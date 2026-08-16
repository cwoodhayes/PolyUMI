"""
ingest/pi_fetch.py - Fetch recorded scenes from a Raspberry Pi over SSH.

Scenes are transferred as tar streams to avoid needing rsync on the Pi.
"""

import logging
import os
import pathlib
import subprocess

log = logging.getLogger(__name__)

REMOTE_RECORDINGS_DIR = '~/recordings'
# Read here rather than per-CLI-option so every consumer (pingest fetch, pingest
# debug-latest, the catalog's Fetch button) honours the same env var — the same one
# fr3_session.sh reads for its Pi pane and deploy. Read at import,
# so a change needs a restart — these are all short-lived processes or long-lived servers
# started from the shell that set it.
#
# The fallback is the bare ssh alias fr3_session.sh also defaults to, deliberately: with both
# unset defaults identical, the three tools agree without anyone exporting anything. It has to
# be an alias in your ssh config carrying a User (see docs/pi-provisioning.md), since there is
# no 'pi@' here to supply one.
DEFAULT_HOST = os.environ.get('POLYUMI_PI_HOST') or 'polyumi-pi'


class PiFetch:
    """SSH client for fetching recorded scenes from a Raspberry Pi."""

    def __init__(self, host: str) -> None:
        """Args: host: SSH hostname or address of the Pi."""
        self.host = host

    def list_remote_scenes(self) -> list[str]:
        """Return scene directory names present on the Pi."""
        result = subprocess.run(
            ['ssh', self.host, f'ls {REMOTE_RECORDINGS_DIR}'],
            capture_output=True,
            text=True,
            check=True,
        )
        return [s for name in result.stdout.splitlines() if (s := name.strip()).startswith('scene_')]

    def resolve_latest_scene(self) -> str:
        """Return the name of the most-recently recorded scene on the Pi."""
        result = subprocess.run(
            ['ssh', self.host, f'readlink -f {REMOTE_RECORDINGS_DIR}/latest'],
            capture_output=True,
            text=True,
            check=True,
        )
        return pathlib.Path(result.stdout.strip()).name

    def list_remote_sessions(self, scene_name: str) -> list[str]:
        """
        Return the *finalized* session directory names inside a remote scene.

        A scene stays open on the Pi for the whole ``start-scene`` run, so a fetch can land
        while a session is mid-write. ``metadata.json`` is written at session creation with
        ``duration_s`` still null and only filled in by ``finalize()``, which makes it the
        finished/unfinished marker. ``grep -L`` prints the files that *don't* match, i.e. the
        finished ones, in a single round trip.

        ``check=False`` because two ordinary outcomes exit non-zero: grep exits 1 when every
        session is still recording, and the shell glob fails on a scene with no sessions yet.
        Both mean "nothing to fetch", not an error.
        """
        result = subprocess.run(
            [
                'ssh',
                self.host,
                f'grep -L \'"duration_s": null\' {REMOTE_RECORDINGS_DIR}/{scene_name}/session_*/metadata.json',
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        names = [pathlib.PurePosixPath(line).parent.name for raw in result.stdout.splitlines() if (line := raw.strip())]
        return sorted(n for n in names if n.startswith('session_'))

    def missing_sessions(self, scene_name: str, local_scene: pathlib.Path) -> list[str]:
        """
        Return finalized remote sessions of ``scene_name`` that aren't in ``local_scene`` yet.

        The unit of "already fetched" is the session, not the scene: a scene grows as episodes
        are recorded into it, so a scene directory existing locally says nothing about whether
        it is complete.

        A local session counts as fetched only once it has a ``metadata.json``. tar extracts in
        place, so a transfer killed part-way — a dropped ssh connection, Ctrl-C, the Pi falling
        off the network — leaves a directory holding some of a session. Taking mere existence as
        proof would strand that session as permanently "already fetched", silently truncated in
        every export built from it. Re-fetching is cheap and tar overwrites, so erring towards
        re-fetching is the safe direction.
        """
        local: set[str] = set()
        if local_scene.is_dir():
            local = {
                p.name
                for p in local_scene.iterdir()
                if p.is_dir() and p.name.startswith('session_') and (p / 'metadata.json').exists()
            }
        return [s for s in self.list_remote_sessions(scene_name) if s not in local]

    def copy_scene(
        self,
        scene_name: str,
        local_path: pathlib.Path,
        verbose: bool = False,
    ) -> None:
        """Copy a named scene directory from the Pi using tar streamed over SSH."""
        self._copy_members([scene_name], local_path.parent, verbose=verbose)

    def copy_sessions(
        self,
        scene_name: str,
        session_names: list[str],
        local_parent: pathlib.Path,
        verbose: bool = False,
    ) -> None:
        """
        Copy individual sessions of a scene, merging into any local copy of that scene.

        ``tar -x`` creates the scene directory if absent and only writes the members in the
        stream, so host-only files — ``scene.json``, ``scene.zarr``, ``slam_logs/``, the SLAM
        atlas, and already-fetched sessions — are left untouched.
        """
        if not session_names:
            return
        self._copy_members([f'{scene_name}/{s}' for s in session_names], local_parent, verbose=verbose)

    def _copy_members(
        self,
        members: list[str],
        local_parent: pathlib.Path,
        verbose: bool = False,
    ) -> None:
        """Stream the given paths (relative to the Pi's recordings dir) into ``local_parent``."""
        local_parent = local_parent.resolve()
        local_parent.mkdir(parents=True, exist_ok=True)

        remote_cmd = [
            'ssh',
            self.host,
            'tar',
            '-C',
            REMOTE_RECORDINGS_DIR,
            '-cf',
            '-',
            *members,
        ]
        extract_cmd = ['tar', '-C', str(local_parent), '-xf', '-']

        if verbose:
            extract_cmd.insert(1, '-v')

        remote_proc = subprocess.Popen(remote_cmd, stdout=subprocess.PIPE)
        if remote_proc.stdout is None:
            raise RuntimeError('Failed to open ssh stream for tar transfer.')

        extract_result = subprocess.run(
            extract_cmd,
            stdin=remote_proc.stdout,
            check=False,
        )
        remote_proc.stdout.close()
        remote_rc = remote_proc.wait()

        if remote_rc != 0:
            raise RuntimeError(f'ssh/tar sender failed with code {remote_rc}')
        if extract_result.returncode != 0:
            raise RuntimeError(f'tar extract failed with code {extract_result.returncode}')

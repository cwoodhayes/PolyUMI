"""
ingest/pi_fetch.py - Fetch recorded scenes from a Raspberry Pi over SSH.

Scenes are transferred as tar streams to avoid needing rsync on the Pi.
"""

import logging
import os
import pathlib
import subprocess
from collections.abc import Iterable

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


def _local_sessions(scene_dir: pathlib.Path) -> set[str]:
    """
    Return the session directories already fetched into ``scene_dir``.

    A local session counts as fetched only once it has a ``metadata.json``. tar extracts in
    place, so a transfer killed part-way — a dropped ssh connection, Ctrl-C, the Pi falling
    off the network — leaves a directory holding some of a session. Taking mere existence as
    proof would strand that session as permanently "already fetched", silently truncated in
    every export built from it. Re-fetching is cheap and tar overwrites, so erring towards
    re-fetching is the safe direction.
    """
    if not scene_dir.is_dir():
        return set()
    return {
        p.name
        for p in scene_dir.iterdir()
        if p.is_dir() and p.name.startswith('session_') and (p / 'metadata.json').exists()
    }


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

    def list_remote_sessions(self) -> dict[str, list[str]]:
        """
        Return ``{scene_name: [finalized session dirs]}`` for every scene on the Pi.

        A scene stays open on the Pi for the whole ``start-scene`` run, so a fetch can land
        while a session is mid-write. ``metadata.json`` is written at session creation with
        ``duration_s`` still null and only filled in by ``finalize()``, which makes it the
        finished/unfinished marker.

        The whole tree in one round trip, not one ssh per scene: this runs before anything is
        transferred, and a Pi holding fifty scenes would otherwise spend fifty handshakes
        deciding there is nothing to do — with the catalog's progress bar sitting at 0
        throughout, since it can't know the total until this returns.

        ``find ... ! -exec grep -q`` rather than ``grep -L``, so the exit status stays worth
        checking: find reports only its own failures and ignores what the predicate matched,
        while grep exits 1 when every session is still recording — indistinguishable from ssh
        failing outright. With ``check=True`` a dead connection or an unreadable directory now
        raises instead of quietly reporting every scene as fully fetched.
        """
        result = subprocess.run(
            [
                'ssh',
                self.host,
                f'find {REMOTE_RECORDINGS_DIR} -mindepth 3 -maxdepth 3 -name metadata.json '
                f'! -exec grep -q \'"duration_s": null\' {{}} \\; -print',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        found: dict[str, list[str]] = {}
        for raw in result.stdout.splitlines():
            if not (line := raw.strip()):
                continue
            path = pathlib.PurePosixPath(line)
            scene, session = path.parent.parent.name, path.parent.name
            if scene.startswith('scene_') and session.startswith('session_'):
                found.setdefault(scene, []).append(session)
        return {scene: sorted(found[scene]) for scene in sorted(found)}

    def missing_sessions(
        self,
        local_recordings: pathlib.Path,
        scene_names: Iterable[str] | None = None,
    ) -> dict[str, list[str]]:
        """
        Return ``{scene: [sessions]}`` finalized on the Pi but not yet under ``local_recordings``.

        ``scene_names`` restricts the answer to those scenes; None means every scene on the Pi.

        The unit of "already fetched" is the session, not the scene: a scene grows as episodes
        are recorded into it, so a scene directory existing locally says nothing about whether
        it is complete. Scenes with nothing outstanding are left out entirely, so the result is
        both the plan and the count.
        """
        wanted = None if scene_names is None else set(scene_names)
        plan: dict[str, list[str]] = {}
        for scene, sessions in self.list_remote_sessions().items():
            if wanted is not None and scene not in wanted:
                continue
            local = _local_sessions(local_recordings / scene)
            if missing := [s for s in sessions if s not in local]:
                plan[scene] = missing
        return plan

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

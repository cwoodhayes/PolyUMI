"""
postprocess/pi_fetch.py - Fetch recorded sessions from a Raspberry Pi over SSH.

Sessions are transferred as tar streams to avoid needing rsync on the Pi.
"""

import logging
import pathlib
import subprocess

log = logging.getLogger(__name__)

REMOTE_RECORDINGS_DIR = '~/recordings'


class PiFetch:
    """SSH client for fetching recorded sessions from a Raspberry Pi."""

    def __init__(self, host: str) -> None:
        """Args: host: SSH hostname or address of the Pi."""
        self.host = host

    def list_remote_sessions(self) -> list[str]:
        """Return session directory names present on the Pi."""
        result = subprocess.run(
            ['ssh', self.host, f'ls {REMOTE_RECORDINGS_DIR}'],
            capture_output=True,
            text=True,
            check=True,
        )
        return [s for name in result.stdout.splitlines() if (s := name.strip()).startswith('session_')]

    def resolve_latest_session(self) -> str:
        """Return the name of the most-recently recorded session on the Pi."""
        result = subprocess.run(
            ['ssh', self.host, f'readlink -f {REMOTE_RECORDINGS_DIR}/latest'],
            capture_output=True,
            text=True,
            check=True,
        )
        return pathlib.Path(result.stdout.strip()).name

    def copy_session(
        self,
        session_name: str,
        local_path: pathlib.Path,
        verbose: bool = False,
    ) -> None:
        """Copy a named session directory from the Pi using tar streamed over SSH."""
        remote_parent = REMOTE_RECORDINGS_DIR
        remote_name = session_name

        local_parent = local_path.parent.resolve()
        local_parent.mkdir(parents=True, exist_ok=True)

        remote_cmd = [
            'ssh',
            self.host,
            'tar',
            '-C',
            remote_parent,
            '-cf',
            '-',
            remote_name,
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

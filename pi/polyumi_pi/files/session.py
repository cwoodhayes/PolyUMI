"""Top-level session file manager."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from polyumi_pi.files.base import SessionDataABC
from polyumi_pi.files.metadata import SessionMetadata

DEFAULT_SESSION_BASE_DIR = pathlib.Path('~/recordings/').expanduser()


@dataclass
class SessionFiles(SessionDataABC):
    """
    Abstraction for a data collection session.

    Currently we expect one-to-one mapping between sessions & demo episodes,
    so we start/stop recording before/after each demo.
    """

    metadata: SessionMetadata

    @classmethod
    def from_file(cls, path: pathlib.Path) -> SessionFiles:
        """Load a session from a session directory."""
        if not path.is_dir():
            raise ValueError(f'Expected session directory, got file: {path}')

        metadata_path = path / 'metadata.json'
        if not metadata_path.is_file():
            raise ValueError(
                f'Metadata file not found at expected path: {metadata_path}'
            )

        metadata = SessionMetadata.from_file(metadata_path)

        return cls(path=path, metadata=metadata)

    @classmethod
    def create(
        cls, base_dir: pathlib.Path = DEFAULT_SESSION_BASE_DIR
    ) -> SessionFiles:
        """Create a new session directory and its associated files."""
        # make a path based on the current ns timestamp

        # kind of annoying but imma give a temporary path since the timestamp
        # is generated in the metadata file and I don't want to duplicate
        # that logic here.
        metadata = SessionMetadata(path=pathlib.Path('/tmp/metadata.json'))
        # using local tz for folder names since this is a human readable
        # helpful name.
        folder_name = metadata.created_at.astimezone().strftime(
            r'session_%Y-%m-%d_%H-%M-%S'
        )
        path = base_dir / folder_name
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

        metadata.path = path / 'metadata.json'

        session = cls(path=path, metadata=metadata)
        session.metadata.to_file()
        return session

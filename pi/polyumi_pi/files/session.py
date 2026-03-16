"""Top-level session file manager."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from polyumi_pi.files.audio import AudioFile
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
    audio: AudioFile | None = None

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

        audio_path = path / 'audio.wav'
        audio = (
            AudioFile.from_file(audio_path) if audio_path.is_file() else None
        )

        return cls(path=path, metadata=metadata, audio=audio)

    @classmethod
    def create(
        cls,
        base_dir: pathlib.Path = DEFAULT_SESSION_BASE_DIR,
        add_latest_symlink: bool = True,
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

        # for convenience
        if add_latest_symlink:
            latest_symlink = base_dir / 'latest'
            if latest_symlink.is_symlink() or latest_symlink.exists():
                latest_symlink.unlink()
            latest_symlink.symlink_to(path)
        return session

    def init_audio(self, sample_rate: int, channels: int, sample_width: int):
        """Create an audio file for this session."""
        if self.audio is not None:
            raise ValueError('Audio file already exists for this session.')

        audio_path = self.path / 'audio.wav'
        self.audio = AudioFile(
            path=audio_path,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )

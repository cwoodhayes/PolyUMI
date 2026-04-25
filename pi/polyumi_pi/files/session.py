"""Top-level session file manager."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone

from polyumi_pi.files.audio import AudioFile
from polyumi_pi.files.base import SessionDataABC
from polyumi_pi.files.metadata import SessionMetadata
from polyumi_pi.files.video import VideoFile

DEFAULT_RECORDINGS_DIR = pathlib.Path('~/recordings/').expanduser()


@dataclass
class SessionFiles(SessionDataABC):
    """
    Abstraction for a data collection session.

    Currently we expect one-to-one mapping between sessions & demo episodes,
    so we start/stop recording before/after each demo.
    """

    metadata: SessionMetadata
    audio: AudioFile | None = None
    video: VideoFile | None = None

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
        audio = AudioFile.from_file(audio_path) if audio_path.is_file() else None

        video_path = path / 'video'
        video = VideoFile.from_file(video_path) if video_path.is_dir() else None

        return cls(path=path, metadata=metadata, audio=audio, video=video)

    @classmethod
    def create(
        cls,
        base_dir: pathlib.Path = DEFAULT_RECORDINGS_DIR,
        add_latest_symlink: bool = True,
        scene_id: str | None = None,
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
            r'session_%Y-%m-%d_%H-%M-%S_' + metadata.session_id[:4]
        )
        path = base_dir / folder_name
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

        metadata.path = path / 'metadata.json'
        if scene_id is not None:
            metadata.scene_id = scene_id

        session = cls(path=path, metadata=metadata)
        session.metadata.to_file()

        # for convenience
        if add_latest_symlink:
            latest_symlink = base_dir / 'latest'
            if latest_symlink.is_symlink() or latest_symlink.exists():
                latest_symlink.unlink()
            latest_symlink.symlink_to(path)
        return session

    def init_audio(
        self,
        sample_rate: int,
        channels: int,
        sample_width: int,
        chunk_ms: int,
    ):
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

        self.metadata.audio_sample_rate = sample_rate
        self.metadata.audio_channels = channels
        self.metadata.audio_chunk_ms = chunk_ms
        self.metadata.n_audio_chunks = 0

    def init_video(self, fps: float, width: int, height: int):
        """Create a video file for this session."""
        if self.video is not None:
            raise ValueError('Video file already exists for this session.')

        video_path = self.path / 'video'
        self.video = VideoFile(
            path=video_path,
            fps=fps,
            width=width,
            height=height,
        )

        self.metadata.camera_fps = int(fps)
        self.metadata.camera_resolution = (width, height)
        self.metadata.n_video_frames = 0

    def set_gopro_sync_time(self, dt: datetime) -> None:
        """Record the datetime used to sync the GoPro clock into session metadata."""
        self.metadata.gopro_sync_time = dt

    def finalize(self):
        """Finalize the session by writing metadata to file."""
        if self.metadata.duration_s is None:
            self.metadata.duration_s = (
                datetime.now(timezone.utc) - self.metadata.created_at
            ).total_seconds()
        self.metadata.to_file()

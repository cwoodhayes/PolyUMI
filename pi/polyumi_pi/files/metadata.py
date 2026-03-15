"""Metadata file abstraction for PolyUMI data collection."""

import json
import pathlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from polyumi_pi.files import base


def _get_git_hash() -> str:
    """Get the current git commit hash."""
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
        cwd=pathlib.Path(__file__).resolve().parent,
    )
    return result.stdout.strip()


_GIT_HASH = _get_git_hash()


@dataclass
class SessionMetadata(base.SessionDataABC):
    """Abstraction for the metadata file recorded during data collection."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_s: float | None = None
    pi_hostname: str | None = None
    camera_fps: int | None = None
    camera_resolution: tuple[int, int] | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    audio_chunk_ms: int | None = None
    n_video_frames: int | None = None
    n_audio_chunks: int | None = None
    video_dropped_frames: int | None = None
    audio_dropped_chunks: int | None = None
    notes: str | None = None
    task: str | None = None
    robot: str | None = None
    polyumi_version: str = field(default_factory=lambda: _GIT_HASH)

    # manually maintained file version to handle breaking changes to
    # the metadata format
    file_version: int = 1

    def __post_init__(self):
        if self.path.name != 'metadata.json':
            raise ValueError(
                f'Expected metadata.json file, got {self.path.name}'
            )
        if self.file_version != 1:
            raise ValueError(
                f'Unsupported metadata file version: {self.file_version}'
            )

    def to_file(self) -> None:
        """Write this metadata to self.path as JSON."""
        data = {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'duration_s': self.duration_s,
            'pi_hostname': self.pi_hostname,
            'camera_fps': self.camera_fps,
            'camera_resolution': (
                list(self.camera_resolution)
                if self.camera_resolution is not None
                else None
            ),
            'audio_sample_rate': self.audio_sample_rate,
            'audio_channels': self.audio_channels,
            'audio_chunk_ms': self.audio_chunk_ms,
            'n_video_frames': self.n_video_frames,
            'n_audio_chunks': self.n_audio_chunks,
            'video_dropped_frames': self.video_dropped_frames,
            'audio_dropped_chunks': self.audio_dropped_chunks,
            'notes': self.notes,
            'task': self.task,
            'robot': self.robot,
            'polyumi_version': self.polyumi_version,
            'file_version': self.file_version,
        }
        self.path.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_file(cls, path: pathlib.Path) -> 'SessionMetadata':
        """Load session metadata from a JSON file."""
        data = json.loads(path.read_text())
        data['path'] = path
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['camera_resolution'] is not None:
            data['camera_resolution'] = tuple(data['camera_resolution'])
        return cls(**data)

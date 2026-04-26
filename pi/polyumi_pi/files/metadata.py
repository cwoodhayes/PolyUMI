"""Metadata file abstraction for PolyUMI data collection."""

import json
import logging
import pathlib
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from polyumi_pi.files import base

log = logging.getLogger('pi_metadata')


def _get_git_hash() -> str:
    """Get the current git commit hash."""
    try:
        from polyumi_pi._version import COMMIT_HASH

        return COMMIT_HASH
    except ImportError as err:
        try:
            # fall back to this if using the git repo without a deployment (i.e. first boot).
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True
            ).strip()
        except Exception:
            raise err


_GIT_HASH = _get_git_hash()


@dataclass
class SessionMetadata(base.SessionDataABC):
    """Abstraction for the metadata file recorded during data collection."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    scene_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_s: float | None = None
    pi_hostname: str = field(default_factory=lambda: socket.gethostname())
    camera_fps: int | None = None
    camera_resolution: tuple[int, int] | None = None
    audio_start_time_ns: int | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    audio_chunk_ms: int | None = None
    n_video_frames: int = 0
    n_audio_chunks: int = 0
    video_dropped_frames: int | None = None
    audio_dropped_chunks: int | None = None
    led_brightness: float | None = None
    gopro_sync_time: datetime | None = None
    first_frame_metadata: dict | None = None
    notes: str | None = None
    task: str | None = None
    robot: str | None = None
    polyumi_version: str = field(default_factory=lambda: _GIT_HASH)

    # manually maintained file version to handle breaking changes to
    # the metadata format
    file_version: int = 1

    def __post_init__(self):
        if self.path.name != 'metadata.json':
            raise ValueError(f'Expected metadata.json file, got {self.path.name}')
        if self.file_version != 1:
            raise ValueError(f'Unsupported metadata file version: {self.file_version}')

    def to_file(self) -> None:
        """Write this metadata to self.path as JSON."""
        data = {
            'session_id': self.session_id,
            'scene_id': self.scene_id,
            'created_at': self.created_at.isoformat(),
            'duration_s': self.duration_s,
            'pi_hostname': self.pi_hostname,
            'camera_fps': self.camera_fps,
            'camera_resolution': (
                list(self.camera_resolution)
                if self.camera_resolution is not None
                else None
            ),
            'audio_start_time_ns': self.audio_start_time_ns,
            'audio_sample_rate': self.audio_sample_rate,
            'audio_channels': self.audio_channels,
            'audio_chunk_ms': self.audio_chunk_ms,
            'n_video_frames': self.n_video_frames,
            'n_audio_chunks': self.n_audio_chunks,
            'video_dropped_frames': self.video_dropped_frames,
            'audio_dropped_chunks': self.audio_dropped_chunks,
            'led_brightness': self.led_brightness,
            'gopro_sync_time': (
                self.gopro_sync_time.isoformat() if self.gopro_sync_time is not None else None
            ),
            'first_frame_metadata': self.first_frame_metadata,
            'notes': self.notes,
            'task': self.task,
            'robot': self.robot,
            'polyumi_version': self.polyumi_version,
            'file_version': self.file_version,
        }
        self.path.write_text(json.dumps(data, indent=2))
        log.info(f'Wrote metadata to {self.path}')

    @classmethod
    def from_file(cls, path: pathlib.Path) -> 'SessionMetadata':
        """Load session metadata from a JSON file."""
        data = json.loads(path.read_text())
        data['path'] = path
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['camera_resolution'] is not None:
            data['camera_resolution'] = tuple(data['camera_resolution'])
        if data.get('gopro_sync_time') is not None:
            data['gopro_sync_time'] = datetime.fromisoformat(data['gopro_sync_time'])
        return cls(**data)

"""SceneFiles: filesystem layout conventions for a pzarr scene directory."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field

from polyumi_pi.files.session import SessionFiles

log = logging.getLogger(__name__)

FINGER_MP4 = 'finger.mp4'
GOPRO_MP4 = 'gopro.mp4'


@dataclass
class SceneFiles:
    """
    Represents the on-disk layout of a scene directory.

    Encodes conventions for where to find sidecar files and the zarr store,
    rather than storing paths inside zarr metadata (which breaks on any move).

    Layout::

        scene_TASKDATE_UUID/
        ├── scene.zarr/
        ├── session_YYYY-MM-DD_hh-mm-ss/
        │   ├── finger.mp4
        │   ├── gopro.mp4
        │   └── ...
        └── scene_TASKDATE_UUID.atlas.osa   (ORB-SLAM3 only)
    """

    path: pathlib.Path
    sessions: list[SessionFiles] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: pathlib.Path) -> SceneFiles:
        """Load a SceneFiles from a scene directory, discovering contained sessions."""
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f'Expected scene directory: {path}')

        sessions: list[SessionFiles] = []
        for child in sorted(path.iterdir()):
            if child.is_dir() and child.name.startswith('session_'):
                try:
                    sessions.append(SessionFiles.from_file(child))
                except Exception as e:
                    log.warning(f'Skipping {child.name}: {e}')

        return cls(path=path, sessions=sessions)

    @staticmethod
    def resolve_zarr_path(path: pathlib.Path) -> pathlib.Path:
        """Accept either a scene directory or a direct zarr path; return the zarr path."""
        path = path.resolve()
        candidate = path / 'scene.zarr'
        return candidate if candidate.exists() else path

    # --- zarr store ---

    @property
    def zarr_path(self) -> pathlib.Path:
        """Path to the pzarr file."""
        return self.path / 'scene.zarr'

    @property
    def zarr_exists(self) -> bool:
        """True if the zarr store exists on disk."""
        return self.zarr_path.exists()

    # --- per-session sidecar accessors ---

    def finger_mp4(self, session: SessionFiles) -> pathlib.Path:
        """Return the conventional path to the finger camera mp4 sidecar for a session."""
        return session.path / FINGER_MP4

    def gopro_mp4(self, session: SessionFiles) -> pathlib.Path:
        """Return the conventional path to the GoPro mp4 sidecar for a session."""
        return session.path / GOPRO_MP4

    # --- scene-level sidecars ---

    @property
    def orb_slam3_atlas(self) -> pathlib.Path:
        """Conventional path for the ORB-SLAM3 persistent atlas sidecar."""
        return self.path / f'{self.path.name}.atlas.osa'

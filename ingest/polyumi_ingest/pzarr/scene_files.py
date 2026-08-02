"""SceneFiles: filesystem layout conventions for a pzarr scene directory."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from polyumi_pi.files.metadata import SessionType
from polyumi_pi.files.session import SessionFiles

if TYPE_CHECKING:
    import zarr

log = logging.getLogger(__name__)

FINGER_MP4 = 'finger.mp4'
GOPRO_MP4 = 'gopro.mp4'


def resolve_gopro_mp4(ep_grp: zarr.Group, scene_zarr: pathlib.Path) -> pathlib.Path:
    """
    Return the original gopro.mp4 sidecar path for an episode group.

    Checks the episode's ``session_dir`` attr first (written by build_pzarr).
    Falls back to matching the episode index against the scene's ``session_*``
    directories sorted by name (the same order build_pzarr uses), for older
    zarrs that predate the attr. Raises FileNotFoundError if not found.

    This is the single source of truth for locating an episode's GoPro footage;
    the SLAM step and the on-demand frame reader both resolve through it.
    """
    scene_dir = scene_zarr.parent
    session_dir_name = ep_grp.attrs.get('session_dir', None)
    if isinstance(session_dir_name, str) and session_dir_name:
        candidate = scene_dir / session_dir_name / GOPRO_MP4
        if candidate.exists():
            return candidate

    ep_key = ep_grp.name.lstrip('/')
    try:
        ep_index = int(ep_key.split('_')[1])
    except (IndexError, ValueError):
        raise FileNotFoundError(f'Could not determine session directory for episode {ep_key!r}')
    session_dirs = sorted(d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith('session_'))
    if ep_index < len(session_dirs):
        candidate = session_dirs[ep_index] / GOPRO_MP4
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'gopro.mp4 not found for {ep_key!r} — expected at '
        f'{session_dirs[ep_index] / GOPRO_MP4 if ep_index < len(session_dirs) else "<no matching session dir>"}'
    )


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
    #: ``{session_dir_name: reason}`` for session directories that failed to load at all — an
    #: empty ``audio.wav``, missing ``metadata.json``. They never become episodes, so
    #: ``build_pzarr`` flags them unusable from here; otherwise they'd be a warning nobody sees
    #: and a scene that quietly has fewer episodes than it has session directories.
    unloadable: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: pathlib.Path) -> SceneFiles:
        """Load a SceneFiles from a scene directory, discovering contained sessions."""
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f'Expected scene directory: {path}')

        sessions: list[SessionFiles] = []
        unloadable: dict[str, str] = {}
        for child in sorted(path.iterdir()):
            if child.is_dir() and child.name.startswith('session_'):
                try:
                    sessions.append(SessionFiles.from_file(child))
                except Exception as e:
                    # !r, not str: a bare EOFError() from wave.open on a zero-byte audio.wav
                    # stringifies to nothing, which used to print `Skipping session_x: `.
                    reason = f'{type(e).__name__}: {e}' if str(e) else repr(e)
                    log.warning(f'Skipping {child.name}: {reason}')
                    unloadable[child.name] = reason

        return cls(path=path, sessions=sessions, unloadable=unloadable)

    @staticmethod
    def resolve_zarr_path(path: pathlib.Path) -> pathlib.Path:
        """Accept either a scene directory or a direct zarr path; return the zarr path."""
        path = path.resolve()
        if path.suffix == '.zarr':
            return path
        return path / 'scene.zarr'

    # --- zarr store ---

    @property
    def zarr_path(self) -> pathlib.Path:
        """Path to the pzarr file."""
        return self.path / 'scene.zarr'

    @property
    def zarr_exists(self) -> bool:
        """True if the zarr store exists on disk."""
        return self.zarr_path.exists()

    # --- session type helpers ---

    @property
    def mapping_session(self) -> SessionFiles | None:
        """Return the MAPPING session for this scene, or None if absent."""
        for s in self.sessions:
            if s.metadata.session_type == SessionType.MAPPING:
                return s
        return None

    @property
    def episode_sessions(self) -> list[SessionFiles]:
        """Return all EPISODE sessions in chronological order."""
        return [s for s in self.sessions if s.metadata.session_type == SessionType.EPISODE]

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

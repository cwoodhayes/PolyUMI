"""Scene file manager: a scene groups one or more recording sessions."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from polyumi_pi.files.base import SessionDataABC
from polyumi_pi.files.session import DEFAULT_RECORDINGS_DIR, SessionFiles

log = logging.getLogger(__name__)


@dataclass
class SceneFiles(SessionDataABC):
    """
    A scene directory that contains one or more session subdirectories.

    All sessions recorded in a single start-scene invocation share the same
    scene_id and live under this directory.  Single-session recordings (e.g.
    record-episode) are wrapped in their own scene directory so the on-disk
    layout is always uniform.
    """

    scene_id: str
    sessions: list[SessionFiles] = field(default_factory=list)
    optitrack_start_time: datetime | None = None
    #: Wall clock at which this scene began, before the first session. Copied into every
    #: session's metadata.json, the only file `pingest fetch` transfers.
    started_at: datetime | None = None

    @classmethod
    def create(
        cls,
        base_dir: pathlib.Path = DEFAULT_RECORDINGS_DIR,
    ) -> SceneFiles:
        """Create a new scene directory under base_dir and update the latest symlink."""
        scene_id = str(uuid4())
        # One `now` for both, so the stamp and the directory name cannot disagree.
        now = datetime.now().astimezone()
        folder_name = now.strftime(r'scene_%Y-%m-%d_%H-%M-%S') + f'_{scene_id[:4]}'
        path = base_dir / folder_name
        return cls(path=path, scene_id=scene_id, started_at=now.astimezone(timezone.utc))

    def create_session(self) -> SessionFiles:
        """Create a new session directory inside this scene."""
        # we create the scene directory here lazily so we don't
        # end up with scenes with no sessions clogging up the recordings directory.
        self.path.mkdir(parents=True, exist_ok=True)

        latest_symlink = self.path.parent / 'latest'
        if latest_symlink.is_symlink() or latest_symlink.exists():
            latest_symlink.unlink()
        latest_symlink.symlink_to(self.path)

        session = SessionFiles.create(
            base_dir=self.path,
            add_latest_symlink=False,
            scene_id=self.scene_id,
        )
        if self.optitrack_start_time is not None:
            session.metadata.optitrack_start_time = self.optitrack_start_time
        session.metadata.scene_started_at = self.started_at
        return session

    @classmethod
    def from_file(cls, path: pathlib.Path) -> SceneFiles:
        """Load a scene from its directory, discovering contained sessions."""
        if not path.is_dir():
            raise ValueError(f'Expected scene directory, got file: {path}')

        sessions: list[SessionFiles] = []
        scene_id = ''
        for child in sorted(path.iterdir()):
            if child.is_dir() and child.name.startswith('session_'):
                try:
                    session = SessionFiles.from_file(child)
                    sessions.append(session)
                    if not scene_id:
                        scene_id = session.metadata.scene_id
                except Exception as err:
                    log.error(f'Error loading session from {child}: {err}')
                    log.exception(err)
                    pass

        # Recovered from the sessions rather than a scene-level file: that is the only place
        # it was ever written (see create_session).
        starts = [s.metadata.scene_started_at for s in sessions if s.metadata.scene_started_at is not None]
        return cls(path=path, scene_id=scene_id, sessions=sessions, started_at=min(starts) if starts else None)

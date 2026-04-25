"""Scene file manager: a scene groups one or more recording sessions."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
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

    @classmethod
    def create(
        cls,
        base_dir: pathlib.Path = DEFAULT_RECORDINGS_DIR,
    ) -> SceneFiles:
        """Create a new scene directory under base_dir and update the latest symlink."""
        scene_id = str(uuid4())
        folder_name = datetime.now().astimezone().strftime(r'scene_%Y-%m-%d_%H-%M-%S') + f'_{scene_id[:4]}'
        path = base_dir / folder_name
        path.mkdir(parents=True, exist_ok=True)

        latest_symlink = base_dir / 'latest'
        if latest_symlink.is_symlink() or latest_symlink.exists():
            latest_symlink.unlink()
        latest_symlink.symlink_to(path)

        return cls(path=path, scene_id=scene_id)

    def create_session(self) -> SessionFiles:
        """Create a new session directory inside this scene."""
        return SessionFiles.create(
            base_dir=self.path,
            add_latest_symlink=False,
            scene_id=self.scene_id,
        )

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

        return cls(path=path, scene_id=scene_id, sessions=sessions)

"""
Read/write helpers for ``scene.json``, the catalog's authoritative scene-level manifest.

Lives in ``ingest`` rather than ``catalog`` because DP export (``export.dp.buffer``) needs to
read it too — to know which episodes are marked unusable — and ``ingest`` owns
preprocessing/export while ``catalog`` only imports it, never
the other way around. ``polyumi_catalog.manifests`` re-exports ``SceneManifest`` from here so
existing catalog call sites are unaffected.

Ingest *writes* this file as well as reading it (``episode_status`` flags a broken episode
unusable), so every mutation goes through :func:`update_scene_manifest` — one read-modify-write
under one lock, shared by the catalog UI's mutations rather than reimplemented beside them.
The lock is process-wide: it serializes concurrent catalog requests and in-process ingest
calls, but not the catalog's ``pingest`` *subprocesses*, which remain last-writer-wins as they
have always been.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

SCENE_MANIFEST_NAME = 'scene.json'

#: Serializes the read-modify-write in `update_scene_manifest`. Without it two near-simultaneous
#: updates to the same scene (marking several episodes unusable back-to-back, say) could
#: interleave and clobber one another, since each rewrites the whole file.
_SCENE_JSON_LOCK = threading.Lock()


@dataclass
class SceneManifest:
    """Authoritative scene-level metadata stored at ``<scene>/scene.json``."""

    scene_id: str
    task: str | None = None
    notes: str | None = None
    unusable_episodes: list[str] = field(default_factory=list)
    #: Per-session DP-export pose source override, keyed by session directory name (same key
    #: space as ``unusable_episodes``). Values are 'optitrack' or 'slam'; a session absent from
    #: this dict exports from its eef.attrs['default_source'] (see EefPoseStep). Written by the
    #: catalog UI's pose-source selector; consumed by export.dp.buffer.
    pose_source_overrides: dict[str, str] = field(default_factory=dict)
    file_version: int = 1

    @classmethod
    def from_scene_dir(cls, scene_dir: pathlib.Path) -> SceneManifest | None:
        """Load the manifest for ``scene_dir``, or return ``None`` if absent."""
        path = scene_dir / SCENE_MANIFEST_NAME
        if not path.is_file():
            return None
        return cls.from_file(path)

    @classmethod
    def from_file(cls, path: pathlib.Path) -> SceneManifest:
        """Load a manifest from an explicit ``scene.json`` path."""
        data = json.loads(path.read_text())
        version = data.get('file_version', 1)
        if version != 1:
            raise ValueError(f'Unsupported scene.json file_version: {version}')
        return cls(
            scene_id=data['scene_id'],
            task=data.get('task'),
            notes=data.get('notes'),
            unusable_episodes=data.get('unusable_episodes', []),
            pose_source_overrides=data.get('pose_source_overrides', {}),
            file_version=version,
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return {
            'scene_id': self.scene_id,
            'task': self.task,
            'notes': self.notes,
            'unusable_episodes': self.unusable_episodes,
            'pose_source_overrides': self.pose_source_overrides,
            'file_version': self.file_version,
        }

    def write_to_scene_dir(self, scene_dir: pathlib.Path) -> pathlib.Path:
        """Write this manifest to ``<scene_dir>/scene.json`` and return the path."""
        path = scene_dir / SCENE_MANIFEST_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _discover_identity(scene_dir: pathlib.Path) -> tuple[str, str | None]:
    """
    Return ``(scene_id, task)`` read from any session's ``metadata.json`` in ``scene_dir``.

    Only used when creating a manifest for a scene that has none — the same fallback
    chain the catalog sync uses, so a manifest written here resolves to the scene row
    the sync would have created anyway rather than forking its identity.
    """
    for session_dir in sorted(scene_dir.glob('session_*')):
        try:
            meta = json.loads((session_dir / 'metadata.json').read_text())
        except (OSError, ValueError):
            continue
        scene_id = meta.get('scene_id')
        if scene_id:
            return str(scene_id), meta.get('task')
    return scene_dir.name, None


@contextlib.contextmanager
def update_scene_manifest(
    scene_dir: pathlib.Path,
    *,
    default_scene_id: str | None = None,
) -> Iterator[SceneManifest]:
    """
    Read ``<scene_dir>/scene.json``, yield it for mutation, and write it back — under one lock.

    The single mutation path for this file: every writer (catalog UI, ingest) goes through
    here, so a partial rewrite can't interleave with another writer's read. The manifest is
    created if absent, since the marker only takes effect once it's on disk — with
    ``default_scene_id`` when the caller already knows the scene's identity (the catalog has
    it on the DB row), otherwise discovered from a session's ``metadata.json``.

    Raising from the body aborts the write, leaving the file untouched.
    """
    with _SCENE_JSON_LOCK:
        manifest = SceneManifest.from_scene_dir(scene_dir)
        if manifest is None:
            if default_scene_id is not None:
                manifest = SceneManifest(scene_id=default_scene_id)
            else:
                scene_id, task = _discover_identity(scene_dir)
                manifest = SceneManifest(scene_id=scene_id, task=task)
        yield manifest
        manifest.write_to_scene_dir(scene_dir)


def set_episode_unusable(
    scene_dir: pathlib.Path,
    session_dir_name: str,
    unusable: bool,
    *,
    default_scene_id: str | None = None,
) -> bool:
    """
    Add or remove ``session_dir_name`` in ``scene.json``'s ``unusable_episodes``.

    Returns True if the set actually changed. Keyed by the session *directory* name rather
    than its session id: directories are immutable once synced, and DP export only has the
    directory name available on the pzarr episode group.
    """
    with update_scene_manifest(scene_dir, default_scene_id=default_scene_id) as manifest:
        before = set(manifest.unusable_episodes)
        after = before | {session_dir_name} if unusable else before - {session_dir_name}
        manifest.unusable_episodes = sorted(after)
        return before != after

"""
Read/write helpers for ``scene.json``, the catalog's authoritative scene-level manifest.

Lives in ``ingest`` rather than ``catalog`` because DP export (``export.dp.buffer``) needs to
read it too — to know which episodes are marked unusable — and ``ingest`` owns
preprocessing/export while ``catalog`` only imports it (docs/catalog-ui-plan.md §10.2), never
the other way around. ``polyumi_catalog.manifests`` re-exports ``SceneManifest`` from here so
existing catalog call sites are unaffected.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

SCENE_MANIFEST_NAME = 'scene.json'


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

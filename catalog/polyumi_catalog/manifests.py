"""
Read/write helpers for the catalog's authoritative on-disk manifests.

``SceneManifest`` (``scene.json`` at a scene root) is the canonical home of a scene's
task assignment, notes, and unusable-episode markers; it lives in ``polyumi_ingest.manifests``
(re-exported here, along with the ``update_scene_manifest`` / ``set_episode_unusable`` writers
every mutation goes through) because DP export needs to read it too, and ingest writes it —
see that module's docstring.
``DatasetManifest`` (``<name>.dataset.json`` beside an exported buffer) records what
scenes/episodes and which code version produced a training dataset.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from polyumi_ingest.manifests import (
    SCENE_MANIFEST_NAME,
    SceneManifest,
    set_episode_unusable,
    update_scene_manifest,
)

__all__ = [
    'SCENE_MANIFEST_NAME',
    'SceneManifest',
    'DatasetMemberSpec',
    'DatasetManifest',
    'set_episode_unusable',
    'update_scene_manifest',
]


@dataclass
class DatasetMemberSpec:
    """One scene's contribution to a dataset (whole scene, or specific episodes)."""

    scene_id: str
    scene_dir: str
    episodes: str | list[int] = 'all'

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return {'scene_id': self.scene_id, 'scene_dir': self.scene_dir, 'episodes': self.episodes}

    @classmethod
    def from_dict(cls, data: dict) -> DatasetMemberSpec:
        """Build a member spec from a manifest dict entry."""
        return cls(scene_id=data['scene_id'], scene_dir=data['scene_dir'], episodes=data.get('episodes', 'all'))


@dataclass
class DatasetManifest:
    """Provenance for an exported dataset, written beside the ``.zarr.zip`` buffer."""

    name: str
    task: str | None = None
    output: str | None = None
    n_episodes: int | None = None
    polyumi_version: str | None = None
    #: Which ``polyumi_ingest.export.dp`` entry point produced this dataset: ``'dp'``
    #: (visuomotor only) or ``'polyumi'`` (adds the contact-mic and finger-camera
    #: modalities). Defaults to ``'dp'``
    #: so manifests written before this field existed still load correctly.
    exporter_type: str = 'dp'
    export_params: dict = field(default_factory=dict)
    members: list[DatasetMemberSpec] = field(default_factory=list)
    #: One record per exported *segment* — a session is cut into several where its trajectory
    #: drops out or teleports, so this is longer than the session count. Carries scene, session,
    #: episode, segment, source, world_frame, n_steps, frame_range, frame_stride, duration_s and
    #: cut_start/cut_end (why the segment begins and ends where it does); see export.dp.buffer's
    #: module docstring. Also embedded in the .zarr.zip's meta attrs; kept here too so it's
    #: readable without opening the buffer. Read fields with .get() — manifests predating a
    #: field simply lack it.
    pose_provenance: list[dict] = field(default_factory=list)
    #: Time accounting from ``polyumi_ingest.timing.dataset_time_totals``, whose module
    #: docstring defines the three. None on manifests written before these fields existed.
    scene_seconds: float | None = None
    episode_seconds: float | None = None
    exported_seconds: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    file_version: int = 1

    @classmethod
    def from_file(cls, path: pathlib.Path) -> DatasetManifest:
        """Load a dataset manifest from a ``*.dataset.json`` path."""
        data = json.loads(path.read_text())
        version = data.get('file_version', 1)
        if version != 1:
            raise ValueError(f'Unsupported dataset manifest file_version: {version}')
        return cls(
            name=data['name'],
            task=data.get('task'),
            output=data.get('output'),
            n_episodes=data.get('n_episodes'),
            polyumi_version=data.get('polyumi_version'),
            exporter_type=data.get('exporter_type', 'dp'),
            export_params=data.get('export_params', {}),
            members=[DatasetMemberSpec.from_dict(m) for m in data.get('members', [])],
            pose_provenance=data.get('pose_provenance', []),
            scene_seconds=data.get('scene_seconds'),
            episode_seconds=data.get('episode_seconds'),
            exported_seconds=data.get('exported_seconds'),
            created_at=datetime.fromisoformat(data['created_at']),
            file_version=version,
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return {
            'name': self.name,
            'task': self.task,
            'created_at': self.created_at.isoformat(),
            'polyumi_version': self.polyumi_version,
            'exporter_type': self.exporter_type,
            'export_params': self.export_params,
            'members': [m.to_dict() for m in self.members],
            'pose_provenance': self.pose_provenance,
            'output': self.output,
            'n_episodes': self.n_episodes,
            'scene_seconds': self.scene_seconds,
            'episode_seconds': self.episode_seconds,
            'exported_seconds': self.exported_seconds,
            'file_version': self.file_version,
        }

    def to_file(self, path: pathlib.Path) -> pathlib.Path:
        """Write this manifest to ``path`` and return it."""
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

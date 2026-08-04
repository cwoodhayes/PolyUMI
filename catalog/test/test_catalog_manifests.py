"""
Round-trip tests for the on-disk manifests, from the catalog side.

Named ``test_catalog_manifests`` rather than ``test_manifests`` so it doesn't collide with
``ingest/test/test_manifests.py``: neither test dir is a package, so pytest imports both by
bare module name and a shared basename makes collecting the two suites together fail.
"""

from __future__ import annotations

import pathlib

import pytest
from polyumi_catalog.manifests import DatasetManifest, DatasetMemberSpec, SceneManifest


def test_scene_manifest_round_trip(tmp_path: pathlib.Path):
    """Writing then reading a scene manifest preserves its fields."""
    scene_dir = tmp_path / 'scene_x'
    scene_dir.mkdir()
    manifest = SceneManifest(scene_id='abc-123', task='fold_towel', notes='left-handed')
    manifest.write_to_scene_dir(scene_dir)

    loaded = SceneManifest.from_scene_dir(scene_dir)
    assert loaded is not None
    assert loaded.scene_id == 'abc-123'
    assert loaded.task == 'fold_towel'
    assert loaded.notes == 'left-handed'


def test_scene_manifest_absent_returns_none(tmp_path: pathlib.Path):
    """A scene dir without scene.json yields None."""
    scene_dir = tmp_path / 'scene_empty'
    scene_dir.mkdir()
    assert SceneManifest.from_scene_dir(scene_dir) is None


def test_scene_manifest_unusable_episodes_round_trip(tmp_path: pathlib.Path):
    """unusable_episodes (session dir names) survives a write/read round trip."""
    scene_dir = tmp_path / 'scene_y'
    scene_dir.mkdir()
    manifest = SceneManifest(scene_id='abc-456', unusable_episodes=['session_1', 'session_3'])
    manifest.write_to_scene_dir(scene_dir)

    loaded = SceneManifest.from_scene_dir(scene_dir)
    assert loaded.unusable_episodes == ['session_1', 'session_3']


def test_scene_manifest_unusable_episodes_defaults_to_empty(tmp_path: pathlib.Path):
    """A scene.json written before this field existed loads with an empty list, not a crash."""
    path = tmp_path / 'scene.json'
    path.write_text('{"scene_id": "abc-789", "file_version": 1}')
    loaded = SceneManifest.from_file(path)
    assert loaded.unusable_episodes == []


def test_scene_manifest_rejects_unknown_version(tmp_path: pathlib.Path):
    """An unsupported file_version raises."""
    path = tmp_path / 'scene.json'
    path.write_text('{"scene_id": "z", "file_version": 999}')
    with pytest.raises(ValueError):
        SceneManifest.from_file(path)


def test_dataset_manifest_round_trip(tmp_path: pathlib.Path):
    """Writing then reading a dataset manifest preserves members and provenance."""
    manifest = DatasetManifest(
        name='fold_towel_v3',
        task='fold_towel',
        output='fold_towel_v3.zarr.zip',
        n_episodes=42,
        polyumi_version='deadbeef',
        export_params={'obs_down_sample_steps': None},
        members=[
            DatasetMemberSpec('s1', 'scene_a/', 'all'),
            DatasetMemberSpec('s2', 'scene_b/', [0, 2, 3]),
        ],
    )
    path = tmp_path / 'fold_towel_v3.dataset.json'
    manifest.to_file(path)

    loaded = DatasetManifest.from_file(path)
    assert loaded.name == 'fold_towel_v3'
    assert loaded.n_episodes == 42
    assert loaded.polyumi_version == 'deadbeef'
    assert len(loaded.members) == 2
    assert loaded.members[1].episodes == [0, 2, 3]


def test_dataset_manifest_pose_provenance_round_trip(tmp_path: pathlib.Path):
    """pose_provenance (per-episode pose-source records) survives a write/read round trip."""
    provenance = [{'scene': 's1', 'session': 'session_0', 'episode': 'episode_0', 'source': 'slam'}]
    manifest = DatasetManifest(name='ds', pose_provenance=provenance)
    path = tmp_path / 'ds.dataset.json'
    manifest.to_file(path)

    loaded = DatasetManifest.from_file(path)
    assert loaded.pose_provenance == provenance


def test_dataset_manifest_pose_provenance_defaults_to_empty(tmp_path: pathlib.Path):
    """A dataset manifest written before this field existed loads with an empty list, not a crash."""
    path = tmp_path / 'ds.dataset.json'
    path.write_text('{"name": "ds", "created_at": "2026-01-01T00:00:00+00:00", "file_version": 1}')
    loaded = DatasetManifest.from_file(path)
    assert loaded.pose_provenance == []

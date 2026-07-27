"""Round-trip tests for the on-disk manifests."""

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

"""Round-trip tests for SceneManifest, the canonical home of scene.json (also read by DP export)."""

from __future__ import annotations

import pathlib

from polyumi_ingest.manifests import SceneManifest


def test_scene_manifest_round_trip(tmp_path: pathlib.Path):
    """Writing then reading a scene manifest preserves its fields, including unusable_episodes."""
    scene_dir = tmp_path / 'scene_x'
    scene_dir.mkdir()
    manifest = SceneManifest(scene_id='abc-123', task='fold_towel', notes='left-handed', unusable_episodes=['s1'])
    manifest.write_to_scene_dir(scene_dir)

    loaded = SceneManifest.from_scene_dir(scene_dir)
    assert loaded is not None
    assert loaded.scene_id == 'abc-123'
    assert loaded.task == 'fold_towel'
    assert loaded.notes == 'left-handed'
    assert loaded.unusable_episodes == ['s1']


def test_scene_manifest_absent_returns_none(tmp_path: pathlib.Path):
    """A scene dir without scene.json yields None."""
    scene_dir = tmp_path / 'scene_empty'
    scene_dir.mkdir()
    assert SceneManifest.from_scene_dir(scene_dir) is None

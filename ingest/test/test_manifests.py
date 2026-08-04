"""Round-trip tests for SceneManifest, the canonical home of scene.json (also read by DP export)."""

from __future__ import annotations

import pathlib

from polyumi_ingest.manifests import SceneManifest


def test_scene_manifest_round_trip(tmp_path: pathlib.Path):
    """Writing then reading a scene manifest preserves its fields, including unusable_episodes."""
    scene_dir = tmp_path / 'scene_x'
    scene_dir.mkdir()
    manifest = SceneManifest(
        scene_id='abc-123',
        task='fold_towel',
        notes='left-handed',
        unusable_episodes=['s1'],
        pose_source_overrides={'s2': 'slam'},
    )
    manifest.write_to_scene_dir(scene_dir)

    loaded = SceneManifest.from_scene_dir(scene_dir)
    assert loaded is not None
    assert loaded.scene_id == 'abc-123'
    assert loaded.task == 'fold_towel'
    assert loaded.notes == 'left-handed'
    assert loaded.unusable_episodes == ['s1']
    assert loaded.pose_source_overrides == {'s2': 'slam'}


def test_scene_manifest_absent_returns_none(tmp_path: pathlib.Path):
    """A scene dir without scene.json yields None."""
    scene_dir = tmp_path / 'scene_empty'
    scene_dir.mkdir()
    assert SceneManifest.from_scene_dir(scene_dir) is None


def test_scene_manifest_pose_source_overrides_defaults_to_empty(tmp_path: pathlib.Path):
    """A scene.json written before this field existed loads with an empty dict, not a crash."""
    path = tmp_path / 'scene.json'
    path.write_text('{"scene_id": "abc-789", "file_version": 1}')
    loaded = SceneManifest.from_file(path)
    assert loaded.pose_source_overrides == {}


def test_scene_manifest_coerces_malformed_collections(tmp_path: pathlib.Path):
    """
    A hand-edited scene.json with the wrong types loads empty rather than crashing later.

    Both fields are read with .get()/set() by DP export and the catalog sync. A null or a
    list where a dict belongs would otherwise surface as an AttributeError deep in those
    callers, on a value nothing had validated.
    """
    path = tmp_path / 'scene.json'
    path.write_text('{"scene_id": "abc", "unusable_episodes": null, "pose_source_overrides": ["s1"]}')

    loaded = SceneManifest.from_file(path)

    assert loaded.unusable_episodes == []
    assert loaded.pose_source_overrides == {}


def test_scene_manifest_drops_non_string_entries(tmp_path: pathlib.Path):
    """Well-shaped containers holding junk keep the usable entries and drop the rest."""
    path = tmp_path / 'scene.json'
    path.write_text(
        '{"scene_id": "abc", "unusable_episodes": ["s1", 7, null], "pose_source_overrides": {"s2": "slam", "s3": 3}}'
    )

    loaded = SceneManifest.from_file(path)

    assert loaded.unusable_episodes == ['s1']
    assert loaded.pose_source_overrides == {'s2': 'slam'}

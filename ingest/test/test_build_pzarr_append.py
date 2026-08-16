"""
Extending an existing scene.zarr with sessions recorded after it was built.

A scene stays open on the Pi for a whole ``start-scene`` run, so the normal sequence is:
record the mapping walk, fetch, preprocess to check the map, *then* record episodes into the
same scene and fetch again. That second build must not restart from scratch — a rebuild would
throw away the SLAM poses, the per-step marks, and any curation already on the store.
"""

from __future__ import annotations

import pathlib

import numpy as np
import zarr
from polyumi_ingest.manifests import SceneManifest, set_episode_unusable
from polyumi_ingest.pzarr.store import build_pzarr
from test_build_pzarr_faults import _write_session

# Explicit timestamps: episode indices are positional over sessions sorted by created_at, and
# these tests are entirely about what that ordering does when sessions arrive later.
_T0 = '2026-07-29T20:00:00.000000+00:00'
_T1 = '2026-07-29T20:10:00.000000+00:00'
_T2 = '2026-07-29T20:20:00.000000+00:00'
_EARLIER = '2026-07-29T19:00:00.000000+00:00'


def _scene_with_two_sessions(tmp_path: pathlib.Path) -> pathlib.Path:
    scene_dir = tmp_path / 'scene_2026-07-29_16-01-53_2bd6'
    scene_dir.mkdir()
    _write_session(scene_dir, 'session_a', created_at=_T0, session_type='MAPPING')
    _write_session(scene_dir, 'session_b', created_at=_T1)
    build_pzarr(scene_dir, skip_gopro=True)
    return scene_dir


def _mark_as_preprocessed(scene_dir: pathlib.Path) -> None:
    """Stand in for a preprocessing run: scene marks, per-episode marks, and a pose array."""
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='a')
    root.attrs['preprocessing_steps'] = [1, 2]
    for key in ('episode_0', 'episode_1'):
        root[key].attrs['preprocessing_steps'] = [1, 2]
        root[key].require_group('gopro').create_array('slam_poses', data=np.zeros((4, 7)))


def test_append_adds_new_sessions_and_keeps_existing_output(tmp_path: pathlib.Path) -> None:
    """The whole point: new episodes appear, everything already computed survives."""
    scene_dir = _scene_with_two_sessions(tmp_path)
    _mark_as_preprocessed(scene_dir)

    _write_session(scene_dir, 'session_c', created_at=_T2)
    build_pzarr(scene_dir, skip_gopro=True, append=True)

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r')
    assert root.attrs['n_episodes'] == 3
    assert root.attrs['build_complete'] is True
    # Existing episodes keep their index, their SLAM output, and their step marks.
    assert root['episode_0'].attrs['session_dir'] == 'session_a'
    assert root['episode_1'].attrs['session_dir'] == 'session_b'
    assert root['episode_1/gopro/slam_poses'].shape == (4, 7)
    assert root['episode_1'].attrs['preprocessing_steps'] == [1, 2]
    assert root.attrs['preprocessing_steps'] == [1, 2]
    # The new one is present and explicitly unprocessed.
    assert root['episode_2'].attrs['session_dir'] == 'session_c'
    assert root['episode_2'].attrs['preprocessing_steps'] == []
    assert root['episode_2/finger/frames'].shape[0] == 4


def test_append_with_nothing_new_is_a_no_op(tmp_path: pathlib.Path) -> None:
    """Re-running on an unchanged scene must not disturb what's already there."""
    scene_dir = _scene_with_two_sessions(tmp_path)
    _mark_as_preprocessed(scene_dir)

    build_pzarr(scene_dir, skip_gopro=True, append=True)

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r')
    assert root.attrs['n_episodes'] == 2
    assert root['episode_0'].attrs['preprocessing_steps'] == [1, 2]
    assert root['episode_1/gopro/slam_poses'].shape == (4, 7)


def test_append_preserves_curation_on_existing_episodes(tmp_path: pathlib.Path) -> None:
    """
    A hand-set unusable mark survives an append.

    A full rebuild deliberately clears these (it is the "start over" operation), so this is
    the behaviour that would silently regress if append fell back to mode='w'.
    """
    scene_dir = _scene_with_two_sessions(tmp_path)
    set_episode_unusable(scene_dir, 'session_b', True)

    _write_session(scene_dir, 'session_c', created_at=_T2)
    build_pzarr(scene_dir, skip_gopro=True, append=True)

    manifest = SceneManifest.from_scene_dir(scene_dir)
    assert manifest is not None
    assert manifest.unusable_episodes == ['session_b']


def test_a_backdated_session_forces_a_full_rebuild(tmp_path: pathlib.Path) -> None:
    """
    Appending is only sound when new sessions sort after the existing ones.

    A session that predates a built episode would have to renumber its successors, which would
    detach every per-episode result from the index naming it — so the store is rebuilt instead.
    """
    scene_dir = _scene_with_two_sessions(tmp_path)
    _mark_as_preprocessed(scene_dir)

    _write_session(scene_dir, 'session_z', created_at=_EARLIER)
    build_pzarr(scene_dir, skip_gopro=True, append=True)

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r')
    assert root.attrs['n_episodes'] == 3
    # Rebuilt in chronological order, and the stale preprocessing output is gone with it.
    assert root['episode_0'].attrs['session_dir'] == 'session_z'
    assert root['episode_1'].attrs['session_dir'] == 'session_a'
    assert root['episode_2'].attrs['session_dir'] == 'session_b'
    assert 'preprocessing_steps' not in root.attrs


def test_append_on_a_missing_store_builds_from_scratch(tmp_path: pathlib.Path) -> None:
    """append=True on a scene with no store yet is just a normal first build."""
    scene_dir = tmp_path / 'scene_2026-07-29_16-01-53_2bd6'
    scene_dir.mkdir()
    _write_session(scene_dir, 'session_a', created_at=_T0, session_type='MAPPING')

    build_pzarr(scene_dir, skip_gopro=True, append=True)

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r')
    assert root.attrs['n_episodes'] == 1
    assert root.attrs['build_complete'] is True

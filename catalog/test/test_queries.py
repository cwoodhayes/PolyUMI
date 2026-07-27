"""Tests for the read-only view-model queries backing the Phase 1 browser."""

from __future__ import annotations

import pathlib

from polyumi_catalog import queries
from polyumi_catalog.db import get_engine
from polyumi_catalog.manifests import SceneManifest
from polyumi_catalog.sync import sync_recordings
from polyumi_pi.files.metadata import SessionMetadata, SessionType
from sqlmodel import Session as DBSession


def _make_session(scene_dir: pathlib.Path, name: str, *, scene_id: str, session_type: SessionType, task: str | None):
    sd = scene_dir / name
    sd.mkdir(parents=True)
    SessionMetadata(
        path=sd / 'metadata.json',
        scene_id=scene_id,
        session_type=session_type,
        task=task,
        n_video_frames=100,
        video_dropped_frames=3 if session_type == SessionType.EPISODE else 0,
    ).to_file()
    return sd


def _populated_engine(tmp_path: pathlib.Path):
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    scene_dir.mkdir(parents=True)
    SceneManifest(scene_id='scene-1', task='fold_towel').write_to_scene_dir(scene_dir)
    _make_session(scene_dir, 'session_1', scene_id='scene-1', session_type=SessionType.MAPPING, task=None)
    _make_session(scene_dir, 'session_2', scene_id='scene-1', session_type=SessionType.EPISODE, task='fold_towel')

    unassigned_dir = rec / 'scene_2026-07-26_11-00-00_efgh'
    unassigned_dir.mkdir(parents=True)
    _make_session(unassigned_dir, 'session_1', scene_id='scene-2', session_type=SessionType.EPISODE, task=None)

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    return engine


def test_list_tasks_includes_pseudo_rows_and_counts(tmp_path: pathlib.Path):
    """The Tasks column always includes All/Unassigned plus every real task, with counts."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        tasks = queries.list_tasks(db)

    by_key = {t['key']: t for t in tasks}
    assert by_key[queries.FILTER_ALL]['scene_count'] == 2
    assert by_key[queries.FILTER_UNASSIGNED]['scene_count'] == 1
    real = [t for t in tasks if not t['pseudo']]
    assert len(real) == 1
    assert real[0]['name'] == 'fold_towel'
    assert real[0]['scene_count'] == 1


def test_list_scenes_filters_by_task(tmp_path: pathlib.Path):
    """Filtering by a real task id, 'all', and 'unassigned' each return the right scene set."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        tasks = {t['name']: t for t in queries.list_tasks(db) if not t['pseudo']}
        task_key = str(tasks['fold_towel']['id'])

        assigned = queries.list_scenes(db, task_key)
        unassigned = queries.list_scenes(db, queries.FILTER_UNASSIGNED)
        everything = queries.list_scenes(db, queries.FILTER_ALL)

    assert [s['scene_id'] for s in assigned] == ['scene-1']
    assert assigned[0]['episode_count'] == 1
    assert assigned[0]['session_count'] == 2
    assert [s['scene_id'] for s in unassigned] == ['scene-2']
    assert len(everything) == 2


def test_list_sessions_for_scene(tmp_path: pathlib.Path):
    """The Episodes column lists both sessions of a scene with type + dropped-frame info."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        sessions = queries.list_sessions(db, 'scene-1')

    assert {s['session_type'] for s in sessions} == {'MAPPING', 'EPISODE'}
    episode = next(s for s in sessions if s['session_type'] == 'EPISODE')
    assert episode['video_dropped_frames'] == 3


def test_scene_detail_flags_task_conflict(tmp_path: pathlib.Path):
    """A session whose task_meta disagrees with the scene's canonical task shows up in conflicts."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-26_12-00-00_ijkl'
    scene_dir.mkdir(parents=True)
    SceneManifest(scene_id='scene-3', task='fold_towel').write_to_scene_dir(scene_dir)
    _make_session(scene_dir, 'session_1', scene_id='scene-3', session_type=SessionType.EPISODE, task='wipe_table')

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)

    with DBSession(engine) as db:
        detail = queries.scene_detail(db, 'scene-3')

    assert detail['kind'] == 'scene'
    assert detail['task_name'] == 'fold_towel'
    assert len(detail['conflicts']) == 1
    assert detail['conflicts'][0]['task_meta'] == 'wipe_table'


def test_detail_helpers_return_empty_for_missing_ids(tmp_path: pathlib.Path):
    """Unknown ids return the {'kind': 'empty'} sentinel rather than raising."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        assert queries.scene_detail(db, 'nope')['kind'] == 'empty'
        assert queries.session_detail(db, 'nope')['kind'] == 'empty'
        assert queries.dataset_detail(db, 999)['kind'] == 'empty'
        assert queries.task_detail(db, '999')['kind'] == 'empty'

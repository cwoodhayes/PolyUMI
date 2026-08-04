"""Tests for the read-only view-model queries backing the Phase 1 browser."""

from __future__ import annotations

import pathlib
from unittest import mock

import zarr
from polyumi_ingest import quality as iquality

from polyumi_catalog import queries
from polyumi_catalog.db import get_engine
from polyumi_catalog.manifests import SceneManifest
from polyumi_catalog.models import Session
from polyumi_catalog.sync import sync_recordings, sync_scene_quality
from polyumi_pi.files.metadata import SessionMetadata, SessionType
from sqlmodel import Session as DBSession


def _add_slam_quality(
    engine,
    scene_dir: pathlib.Path,
    session_dirname: str,
    *,
    scene_id: str = 'scene-1',
    n_total: int,
    n_lost: int,
    fed: bool = False,
) -> None:
    """
    Stamp a one-episode scene.zarr with SLAM quality attrs for ``session_dirname``.

    Also syncs them into the catalog DB, which is where the queries read them from — the
    real writer (preprocessing) is followed by a sync for the same reason. Writing only to
    pzarr leaves the catalog showing the pre-run numbers.

    ``fed`` additionally writes the post-chirp fed-grid counts. Those are what the
    auto-unusable thresholds judge — without them ``quality._fed_frame_counts`` has
    nothing to go on and every episode reads as usable — while ``tracking_ratio``
    alone is enough for the advisory ``low_quality`` badge.
    """
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 1
    ep = root.require_group('episode_0')
    ep.attrs['session_dir'] = session_dirname
    slam_grp = ep.require_group('annotations').require_group('slam')
    slam_grp.attrs['n_frames_total'] = n_total
    slam_grp.attrs['n_frames_lost'] = n_lost
    slam_grp.attrs['tracking_ratio'] = (n_total - n_lost) / n_total
    slam_grp.attrs['n_relocalization_events'] = 0
    if fed:
        slam_grp.attrs['n_frames_fed_post_chirp'] = n_total
        slam_grp.attrs['n_frames_fed_lost_post_chirp'] = n_lost
    sync_scene_quality(scene_id, engine)


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


def _mark_unusable(engine, scene_id: str) -> None:
    """Manually mark ``scene_id``'s EPISODE session unusable, as the UI's toggle does."""
    with DBSession(engine) as db:
        episode = next(s for s in queries.list_sessions(db, scene_id) if s['session_type'] == 'EPISODE')
        row = db.get(Session, episode['session_id'])
        row.unusable = True
        db.add(row)
        db.commit()


def test_list_scenes_usable_episode_count(tmp_path: pathlib.Path):
    """
    The Scenes column counts usable episodes, excluding both flavours of unusable.

    With no SLAM results at all nothing has ruled the episode out, so it still counts.
    """
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        scene = next(s for s in queries.list_scenes(db, queries.FILTER_ALL) if s['scene_id'] == 'scene-1')
        assert scene['episode_count'] == 1
        assert scene['usable_episode_count'] == 1  # no pzarr yet
        scene_dir = pathlib.Path(scene['dir'])

    _add_slam_quality(engine, scene_dir, 'session_2', n_total=100, n_lost=5, fed=True)  # 95 tracked, 5 lost
    with DBSession(engine) as db:
        scene = next(s for s in queries.list_scenes(db, queries.FILTER_ALL) if s['scene_id'] == 'scene-1')
        assert scene['usable_episode_count'] == 1

    _add_slam_quality(engine, scene_dir, 'session_2', n_total=100, n_lost=60, fed=True)  # 60 lost > max_lost_frames
    with DBSession(engine) as db:
        scene = next(s for s in queries.list_scenes(db, queries.FILTER_ALL) if s['scene_id'] == 'scene-1')
        assert scene['episode_count'] == 1  # total is unchanged
        assert scene['usable_episode_count'] == 0


def test_list_scenes_usable_episode_count_honours_manual_marking(tmp_path: pathlib.Path):
    """A manually-marked episode drops out of the usable count even with clean SLAM."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        scene_dir = pathlib.Path(queries.scene_detail(db, 'scene-1')['dir'])
    _add_slam_quality(engine, scene_dir, 'session_2', n_total=100, n_lost=5, fed=True)
    _mark_unusable(engine, 'scene-1')

    with DBSession(engine) as db:
        scene = next(s for s in queries.list_scenes(db, queries.FILTER_ALL) if s['scene_id'] == 'scene-1')
    assert scene['episode_count'] == 1
    assert scene['usable_episode_count'] == 0


def test_task_detail_usable_episode_count(tmp_path: pathlib.Path):
    """task_detail reports usable episodes alongside the total, per filter."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        tasks = {t['name']: t for t in queries.list_tasks(db) if not t['pseudo']}
        task_key = str(tasks['fold_towel']['id'])
        detail = queries.task_detail(db, queries.FILTER_ALL)
        assert detail['episode_count'] == 2  # scene-1 + scene-2
        assert detail['usable_episode_count'] == 2

    _mark_unusable(engine, 'scene-1')  # scene-1 is the fold_towel one; scene-2 is unassigned
    with DBSession(engine) as db:
        detail = queries.task_detail(db, queries.FILTER_ALL)
        assert detail['episode_count'] == 2
        assert detail['usable_episode_count'] == 1

        assert queries.task_detail(db, task_key)['usable_episode_count'] == 0
        assert queries.task_detail(db, queries.FILTER_UNASSIGNED)['usable_episode_count'] == 1


def test_task_detail_usable_count_follows_the_thresholds_not_a_stored_verdict(tmp_path: pathlib.Path):
    """
    Editing the thresholds reclassifies cached episodes with no re-sync.

    The point of caching only the measurements: the verdict is still policy applied on read.
    """
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        scene_dir = pathlib.Path(queries.scene_detail(db, 'scene-1')['dir'])
    _add_slam_quality(engine, scene_dir, 'session_2', n_total=100, n_lost=5, fed=True)

    with DBSession(engine) as db:
        assert queries.task_detail(db, queries.FILTER_ALL)['usable_episode_count'] == 2

    iquality.load_quality_thresholds.cache_clear()
    try:
        with mock.patch.object(
            iquality, 'load_quality_thresholds', return_value=iquality.QualityThresholds(max_lost_frames=1)
        ):
            with DBSession(engine) as db:
                # same cached numbers, stricter threshold -> scene-1's episode is now excluded
                assert queries.task_detail(db, queries.FILTER_ALL)['usable_episode_count'] == 1
    finally:
        iquality.load_quality_thresholds.cache_clear()


def test_list_sessions_for_scene(tmp_path: pathlib.Path):
    """The Episodes column lists both sessions of a scene with type + dropped-frame info."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        sessions = queries.list_sessions(db, 'scene-1')

    assert {s['session_type'] for s in sessions} == {'MAPPING', 'EPISODE'}
    episode = next(s for s in sessions if s['session_type'] == 'EPISODE')
    assert episode['video_dropped_frames'] == 3


def test_list_sessions_and_detail_include_unusable(tmp_path: pathlib.Path):
    """Both list_sessions and session_detail expose the unusable flag once a session is marked."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        sessions = queries.list_sessions(db, 'scene-1')
        episode = next(s for s in sessions if s['session_type'] == 'EPISODE')
        assert episode['unusable'] is False
        assert queries.session_detail(db, episode['session_id'])['unusable'] is False

        row = db.get(Session, episode['session_id'])
        row.unusable = True
        db.add(row)
        db.commit()

    with DBSession(engine) as db:
        sessions = queries.list_sessions(db, 'scene-1')
        episode = next(s for s in sessions if s['session_type'] == 'EPISODE')
        assert episode['unusable'] is True
        assert queries.session_detail(db, episode['session_id'])['unusable'] is True


def test_session_detail_includes_notes(tmp_path: pathlib.Path):
    """session_detail exposes the session's notes, synced from its metadata.json."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        sessions = queries.list_sessions(db, 'scene-1')
        episode_id = next(s['session_id'] for s in sessions if s['session_type'] == 'EPISODE')
        assert queries.session_detail(db, episode_id)['notes'] is None

        row = db.get(Session, episode_id)
        row.notes = 'gripper slipped'
        db.add(row)
        db.commit()

    with DBSession(engine) as db:
        assert queries.session_detail(db, episode_id)['notes'] == 'gripper slipped'


def test_list_sessions_includes_slam_quality(tmp_path: pathlib.Path):
    """Episodes with SLAM results carry a tracking ratio and low_quality flag in the list."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        scene = queries.scene_detail(db, 'scene-1')
    scene_dir = pathlib.Path(scene['dir'])
    _add_slam_quality(engine, scene_dir, 'session_2', n_total=100, n_lost=60)  # 40% tracked -> low quality

    with DBSession(engine) as db:
        sessions = queries.list_sessions(db, 'scene-1')

    episode = next(s for s in sessions if s['session_type'] == 'EPISODE')
    mapping = next(s for s in sessions if s['session_type'] == 'MAPPING')
    assert episode['slam_tracking_ratio'] == 0.4
    assert episode['slam_low_quality'] is True
    assert mapping['slam_tracking_ratio'] is None  # no matching episode group for this session


def test_session_detail_includes_slam_quality_when_available(tmp_path: pathlib.Path):
    """The session detail panel exposes SLAM stats once pzarr + SLAM results exist."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        scene = queries.scene_detail(db, 'scene-1')
        sessions = queries.list_sessions(db, 'scene-1')
        session_id = next(s['session_id'] for s in sessions if s['session_type'] == 'EPISODE')
    scene_dir = pathlib.Path(scene['dir'])

    with DBSession(engine) as db:
        assert queries.session_detail(db, session_id)['slam'] is None  # no pzarr yet

    _add_slam_quality(engine, scene_dir, 'session_2', n_total=50, n_lost=5)
    with DBSession(engine) as db:
        detail = queries.session_detail(db, session_id)
    assert detail['slam']['n_frames_lost'] == 5
    assert detail['slam']['tracking_ratio'] == 0.9


def test_session_detail_includes_gopro_fps(tmp_path: pathlib.Path, monkeypatch):
    """
    session_detail exposes gopro_fps regardless of pzarr build state.

    Unlike slam/pzarr_streams, this reads gopro.mp4 directly (via thumbnails.gopro_fps),
    so it should be available even before any pzarr has been built for the scene.
    """
    monkeypatch.setattr('polyumi_catalog.thumbnails.gopro_fps', lambda session_dir: 59.94)

    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        sessions = queries.list_sessions(db, 'scene-1')
        session_id = next(s['session_id'] for s in sessions if s['session_type'] == 'EPISODE')
        detail = queries.session_detail(db, session_id)

    assert detail['pzarr_exists'] is False  # no pzarr built yet
    assert detail['gopro_fps'] == 59.94


def test_session_detail_includes_pzarr_streams_when_available(tmp_path: pathlib.Path):
    """The session detail panel exposes a stream shape/rate table once pzarr exists."""
    import numpy as np

    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        scene = queries.scene_detail(db, 'scene-1')
        sessions = queries.list_sessions(db, 'scene-1')
        session_id = next(s['session_id'] for s in sessions if s['session_type'] == 'EPISODE')
    scene_dir = pathlib.Path(scene['dir'])

    with DBSession(engine) as db:
        assert queries.session_detail(db, session_id)['pzarr_streams'] is None  # no pzarr yet

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 1
    ep = root.require_group('episode_0')
    ep.attrs['session_dir'] = 'session_2'
    finger_grp = ep.require_group('finger')
    finger_grp.create_array('frames', data=np.zeros((5, 4, 4, 3), dtype='uint8'))
    ts_grp = ep.require_group('timestamps')
    ts_grp.create_array('finger', data=np.linspace(0.0, 1.0, 5))
    ann_grp = ep.require_group('annotations')
    ann_grp.attrs['episode_start'] = 0.0
    ann_grp.attrs['episode_end'] = 1.0

    with DBSession(engine) as db:
        detail = queries.session_detail(db, session_id)
    assert detail['pzarr_streams']['episode_index'] == 0
    labels = {s['label'] for s in detail['pzarr_streams']['streams']}
    assert 'finger/frames' in labels


def test_session_detail_exposes_pose_source_override_and_available_sources(tmp_path: pathlib.Path):
    """session_detail surfaces the DB-cached override and the pzarr's available_sources."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        scene = queries.scene_detail(db, 'scene-1')
        sessions = queries.list_sessions(db, 'scene-1')
        session_id = next(s['session_id'] for s in sessions if s['session_type'] == 'EPISODE')
    scene_dir = pathlib.Path(scene['dir'])

    with DBSession(engine) as db:
        detail = queries.session_detail(db, session_id)
    assert detail['pose_source_override'] is None
    assert detail['available_pose_sources'] is None  # no pzarr yet — unknown, not empty

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 1
    ep = root.require_group('episode_0')
    ep.attrs['session_dir'] = 'session_2'
    eef_grp = ep.require_group('eef')
    eef_grp.attrs['available_sources'] = ['optitrack', 'slam']

    with DBSession(engine) as db:
        detail = queries.session_detail(db, session_id)
    assert detail['available_pose_sources'] == ['optitrack', 'slam']

    with DBSession(engine) as db:
        row = db.get(Session, session_id)
        row.pose_source_override = 'slam'
        db.add(row)
        db.commit()

    with DBSession(engine) as db:
        assert queries.session_detail(db, session_id)['pose_source_override'] == 'slam'


def test_scene_detail_includes_quality_summary(tmp_path: pathlib.Path):
    """scene_detail aggregates SLAM quality and total dropped video frames across sessions."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        scene = queries.scene_detail(db, 'scene-1')
    scene_dir = pathlib.Path(scene['dir'])
    assert scene['quality'] == {
        'n_episodes_with_slam': 0,
        'avg_tracking_ratio': None,
        'n_low_quality': 0,
        'n_auto_unusable': 0,
    }
    assert scene['total_dropped_video_frames'] == 3  # from the EPISODE session's metadata

    _add_slam_quality(engine, scene_dir, 'session_2', n_total=100, n_lost=10)
    with DBSession(engine) as db:
        scene = queries.scene_detail(db, 'scene-1')
    assert scene['quality']['n_episodes_with_slam'] == 1
    assert scene['quality']['avg_tracking_ratio'] == 0.9
    assert scene['quality']['n_low_quality'] == 0


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


def test_task_detail_includes_episode_count(tmp_path: pathlib.Path):
    """task_detail reports the EPISODE-session count across a task's scenes, for real and pseudo rows alike."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        tasks = {t['name']: t for t in queries.list_tasks(db) if not t['pseudo']}
        task_id = tasks['fold_towel']['id']

        real = queries.task_detail(db, str(task_id))
        all_scenes = queries.task_detail(db, queries.FILTER_ALL)
        unassigned = queries.task_detail(db, queries.FILTER_UNASSIGNED)

    # scene-1 (fold_towel) has 1 EPISODE session (session_2; session_1 is MAPPING).
    assert real['episode_count'] == 1
    # scene-2 (unassigned) has 1 EPISODE session.
    assert unassigned['episode_count'] == 1
    # both scenes' EPISODE sessions combined.
    assert all_scenes['episode_count'] == 2


def test_detail_helpers_return_empty_for_missing_ids(tmp_path: pathlib.Path):
    """Unknown ids return the {'kind': 'empty'} sentinel rather than raising."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        assert queries.scene_detail(db, 'nope')['kind'] == 'empty'
        assert queries.session_detail(db, 'nope')['kind'] == 'empty'
        assert queries.dataset_detail(db, 999)['kind'] == 'empty'
        assert queries.task_detail(db, '999')['kind'] == 'empty'


def test_list_scene_options_includes_task_name(tmp_path: pathlib.Path):
    """Scene options for the dataset builder carry a friendly name and their task's name."""
    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        options = queries.list_scene_options(db)

    by_id = {o['scene_id']: o for o in options}
    assert by_id['scene-1']['task_name'] == 'fold_towel'
    assert by_id['scene-2']['task_name'] is None


def test_dataset_detail_lists_member_scenes(tmp_path: pathlib.Path):
    """dataset_detail resolves each DatasetMember's scene_id to a friendly name."""
    from polyumi_catalog.models import Dataset, DatasetMember

    engine = _populated_engine(tmp_path)
    with DBSession(engine) as db:
        dataset = Dataset(name='fold_towel_v1', n_episodes=3)
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        db.add(DatasetMember(dataset_id=dataset.id, scene_id='scene-1', episodes='all'))
        db.commit()

        detail = queries.dataset_detail(db, dataset.id)

    assert detail['kind'] == 'dataset'
    assert len(detail['members']) == 1
    assert detail['members'][0]['scene_id'] == 'scene-1'
    assert detail['members'][0]['episodes'] == 'all'
    assert detail['members'][0]['name']  # resolved to the scene's basename, not just the id

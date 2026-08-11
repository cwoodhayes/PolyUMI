"""
Tests for Phase 2 mutations: create/rename tasks and scene->task assignment.

The Phase 2 deliverable is: "retag a scene, drop the
DB, re-sync, tag survives" — i.e. every mutation here must write scene.json
(authoritative), not just the DB row. Each test below verifies that by dropping and
rebuilding the DB from a fresh engine and re-running sync.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest
from polyumi_catalog.db import get_engine
from polyumi_catalog.manifests import SceneManifest
from polyumi_catalog.models import DatasetMember, Scene, Session, Task
from polyumi_catalog.mutations import (
    MutationError,
    assign_scene_task,
    create_task,
    delete_scene,
    rename_task,
    set_scene_notes,
    set_session_notes,
    set_session_pose_source,
    set_session_unusable,
    set_task_description,
)
from polyumi_catalog.sync import sync_recordings
from polyumi_pi.files.metadata import SessionMetadata, SessionType
from sqlmodel import Session as DBSession
from sqlmodel import select


def _make_scene(rec: pathlib.Path, dirname: str, *, scene_id: str, task: str | None) -> pathlib.Path:
    scene_dir = rec / dirname
    scene_dir.mkdir(parents=True)
    if task is not None:
        SceneManifest(scene_id=scene_id, task=task).write_to_scene_dir(scene_dir)
    sd = scene_dir / 'session_1'
    sd.mkdir()
    SessionMetadata(path=sd / 'metadata.json', scene_id=scene_id, session_type=SessionType.EPISODE).to_file()
    return scene_dir


def test_create_task_is_idempotent_by_name(tmp_path: pathlib.Path):
    """Creating a task with an existing name returns the existing row, not a duplicate."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        t1 = create_task(db, 'fold_towel')
        t2 = create_task(db, 'fold_towel')
        assert t1.id == t2.id
        assert len(db.exec(select(Task)).all()) == 1


def test_create_task_rejects_empty_name(tmp_path: pathlib.Path):
    """An empty/whitespace name is rejected rather than silently creating a blank task."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        with pytest.raises(MutationError):
            create_task(db, '   ')


def test_assign_scene_task_writes_scene_json_and_survives_resync(tmp_path: pathlib.Path):
    """Assigning a scene's task rewrites scene.json; a dropped-and-resynced DB recovers it."""
    rec = tmp_path / 'recordings'
    _make_scene(rec, 'scene_2026-07-26_10-00-00_abcd', scene_id='scene-1', task=None)

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        task = create_task(db, 'fold_towel')
        assign_scene_task(db, 'scene-1', task.id)
        assert db.get(Scene, 'scene-1').task_id == task.id

    manifest = SceneManifest.from_scene_dir(rec / 'scene_2026-07-26_10-00-00_abcd')
    assert manifest is not None
    assert manifest.task == 'fold_towel'

    # simulate a full DB rebuild
    rebuilt = get_engine(tmp_path / 'catalog2.db')
    sync_recordings(rec, rebuilt)
    with DBSession(rebuilt) as db:
        scene = db.get(Scene, 'scene-1')
        task = db.get(Task, scene.task_id)
        assert task.name == 'fold_towel'


def test_assign_scene_task_can_unassign(tmp_path: pathlib.Path):
    """Passing task_id=None clears both the DB row and scene.json's task field."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-26_11-00-00_efgh', scene_id='scene-2', task='fold_towel')

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        assign_scene_task(db, 'scene-2', None)
        assert db.get(Scene, 'scene-2').task_id is None

    manifest = SceneManifest.from_scene_dir(scene_dir)
    assert manifest.task is None


def test_assign_scene_task_rejects_unknown_ids(tmp_path: pathlib.Path):
    """Unknown scene or task ids raise MutationError instead of corrupting state."""
    rec = tmp_path / 'recordings'
    _make_scene(rec, 'scene_2026-07-26_12-00-00_ijkl', scene_id='scene-3', task=None)
    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)

    with DBSession(engine) as db:
        with pytest.raises(MutationError):
            assign_scene_task(db, 'no-such-scene', None)
        with pytest.raises(MutationError):
            assign_scene_task(db, 'scene-3', 999)


def test_set_session_unusable_writes_scene_json_and_survives_resync(tmp_path: pathlib.Path):
    """Marking a session unusable rewrites scene.json; a dropped-and-resynced DB recovers it."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-26_18-00-00_yzgh', scene_id='scene-7', task=None)

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        session_id = db.exec(select(Session).where(Session.scene_id == 'scene-7')).first().session_id
        set_session_unusable(db, session_id, True)
        assert db.get(Session, session_id).unusable is True

    manifest = SceneManifest.from_scene_dir(scene_dir)
    assert manifest.unusable_episodes == ['session_1']

    rebuilt = get_engine(tmp_path / 'catalog2.db')
    sync_recordings(rec, rebuilt)
    with DBSession(rebuilt) as db:
        row = db.exec(select(Session).where(Session.scene_id == 'scene-7')).first()
        assert row.unusable is True


def test_set_session_unusable_can_clear(tmp_path: pathlib.Path):
    """Marking a session usable again removes it from scene.json's unusable_episodes."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-26_19-00-00_yzij', scene_id='scene-8', task=None)
    SceneManifest(scene_id='scene-8', unusable_episodes=['session_1']).write_to_scene_dir(scene_dir)

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        session_id = db.exec(select(Session).where(Session.scene_id == 'scene-8')).first().session_id
        assert db.get(Session, session_id).unusable is True
        set_session_unusable(db, session_id, False)
        assert db.get(Session, session_id).unusable is False

    manifest = SceneManifest.from_scene_dir(scene_dir)
    assert manifest.unusable_episodes == []


def test_set_session_unusable_rejects_unknown_session(tmp_path: pathlib.Path):
    """An unknown session_id raises MutationError instead of silently no-op'ing."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        with pytest.raises(MutationError):
            set_session_unusable(db, 'no-such-session', True)


def test_set_session_pose_source_writes_scene_json_and_survives_resync(tmp_path: pathlib.Path):
    """Setting a pose-source override rewrites scene.json; a dropped-and-resynced DB recovers it."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-28_10-00-00_pose1', scene_id='scene-pose-1', task=None)

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        session_id = db.exec(select(Session).where(Session.scene_id == 'scene-pose-1')).first().session_id
        set_session_pose_source(db, session_id, 'slam')
        assert db.get(Session, session_id).pose_source_override == 'slam'

    manifest = SceneManifest.from_scene_dir(scene_dir)
    assert manifest.pose_source_overrides == {'session_1': 'slam'}

    rebuilt = get_engine(tmp_path / 'catalog2.db')
    sync_recordings(rec, rebuilt)
    with DBSession(rebuilt) as db:
        row = db.exec(select(Session).where(Session.scene_id == 'scene-pose-1')).first()
        assert row.pose_source_override == 'slam'


def test_set_session_pose_source_default_clears_override(tmp_path: pathlib.Path):
    """Setting the source back to 'default' removes the scene.json entry entirely."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-28_11-00-00_pose2', scene_id='scene-pose-2', task=None)
    SceneManifest(scene_id='scene-pose-2', pose_source_overrides={'session_1': 'optitrack'}).write_to_scene_dir(
        scene_dir
    )

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        session_id = db.exec(select(Session).where(Session.scene_id == 'scene-pose-2')).first().session_id
        assert db.get(Session, session_id).pose_source_override == 'optitrack'
        set_session_pose_source(db, session_id, 'default')
        assert db.get(Session, session_id).pose_source_override is None

    manifest = SceneManifest.from_scene_dir(scene_dir)
    assert manifest.pose_source_overrides == {}


def test_set_session_pose_source_rejects_unknown_session(tmp_path: pathlib.Path):
    """An unknown session_id raises MutationError instead of silently no-op'ing."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        with pytest.raises(MutationError):
            set_session_pose_source(db, 'no-such-session', 'slam')


def test_set_session_pose_source_rejects_unknown_value(tmp_path: pathlib.Path):
    """A source outside {'default', 'optitrack', 'slam'} is rejected before touching any state."""
    rec = tmp_path / 'recordings'
    _make_scene(rec, 'scene_2026-07-28_12-00-00_pose3', scene_id='scene-pose-3', task=None)
    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        session_id = db.exec(select(Session).where(Session.scene_id == 'scene-pose-3')).first().session_id
        with pytest.raises(MutationError):
            set_session_pose_source(db, session_id, 'lidar')


def test_set_session_pose_source_rejects_source_episode_cannot_supply(tmp_path: pathlib.Path):
    """Overriding to a source the episode's pzarr never computed (no slam here) is rejected."""
    import zarr

    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-28_13-00-00_pose4', scene_id='scene-pose-4', task=None)
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w', zarr_format=2)
    root.attrs['n_episodes'] = 1
    ep = root.create_group('episode_0')
    ep.attrs['session_dir'] = 'session_1'
    eef = ep.create_group('eef')
    eef.attrs['available_sources'] = ['optitrack']  # no slam

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        session_id = db.exec(select(Session).where(Session.scene_id == 'scene-pose-4')).first().session_id
        with pytest.raises(MutationError, match='optitrack'):
            set_session_pose_source(db, session_id, 'slam')
        # optitrack IS available, so that override succeeds
        set_session_pose_source(db, session_id, 'optitrack')
        assert db.get(Session, session_id).pose_source_override == 'optitrack'


def test_set_scene_notes_writes_scene_json_and_survives_resync(tmp_path: pathlib.Path):
    """Setting a scene's notes rewrites scene.json; a dropped-and-resynced DB recovers it."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-27_10-00-00_yzkl', scene_id='scene-9', task=None)

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        set_scene_notes(db, 'scene-9', 'left-handed grasp')
        assert db.get(Scene, 'scene-9').notes == 'left-handed grasp'

    manifest = SceneManifest.from_scene_dir(scene_dir)
    assert manifest.notes == 'left-handed grasp'

    rebuilt = get_engine(tmp_path / 'catalog2.db')
    sync_recordings(rec, rebuilt)
    with DBSession(rebuilt) as db:
        assert db.get(Scene, 'scene-9').notes == 'left-handed grasp'


def test_set_scene_notes_blank_clears(tmp_path: pathlib.Path):
    """A blank/whitespace-only value clears notes rather than storing an empty string."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-27_11-00-00_yzmn', scene_id='scene-10', task=None)
    SceneManifest(scene_id='scene-10', notes='old note').write_to_scene_dir(scene_dir)

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        set_scene_notes(db, 'scene-10', '   ')
        assert db.get(Scene, 'scene-10').notes is None
    assert SceneManifest.from_scene_dir(scene_dir).notes is None


def test_set_scene_notes_rejects_unknown_scene(tmp_path: pathlib.Path):
    """An unknown scene_id raises MutationError instead of a crash."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        with pytest.raises(MutationError):
            set_scene_notes(db, 'no-such-scene', 'hello')


def test_set_session_notes_writes_metadata_json_and_survives_resync(tmp_path: pathlib.Path):
    """Setting a session's notes rewrites its metadata.json, preserving other fields."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-27_12-00-00_yzop', scene_id='scene-11', task=None)
    sd = scene_dir / 'session_1'

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        session_id = db.exec(select(Session).where(Session.scene_id == 'scene-11')).first().session_id
        set_session_notes(db, session_id, 'gripper slipped mid-grasp')
        assert db.get(Session, session_id).notes == 'gripper slipped mid-grasp'

    meta = SessionMetadata.from_file(sd / 'metadata.json')
    assert meta.notes == 'gripper slipped mid-grasp'
    assert meta.session_type == SessionType.EPISODE  # other fields survive the rewrite
    assert meta.scene_id == 'scene-11'

    rebuilt = get_engine(tmp_path / 'catalog2.db')
    sync_recordings(rec, rebuilt)
    with DBSession(rebuilt) as db:
        row = db.exec(select(Session).where(Session.scene_id == 'scene-11')).first()
        assert row.notes == 'gripper slipped mid-grasp'


def test_set_session_notes_blank_clears(tmp_path: pathlib.Path):
    """A blank/whitespace-only value clears notes rather than storing an empty string."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-27_13-00-00_yzqr', scene_id='scene-12', task=None)
    sd = scene_dir / 'session_1'

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        session_id = db.exec(select(Session).where(Session.scene_id == 'scene-12')).first().session_id
        set_session_notes(db, session_id, 'a note')
        set_session_notes(db, session_id, '   ')
        assert db.get(Session, session_id).notes is None
    assert SessionMetadata.from_file(sd / 'metadata.json').notes is None


def test_set_session_notes_rejects_unknown_session(tmp_path: pathlib.Path):
    """An unknown session_id raises MutationError instead of a crash."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        with pytest.raises(MutationError):
            set_session_notes(db, 'no-such-session', 'hello')


def test_set_session_notes_rejects_missing_metadata_json(tmp_path: pathlib.Path):
    """A synced session whose metadata.json has since vanished from disk raises MutationError, not FileNotFoundError."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-27_14-00-00_yzst', scene_id='scene-13', task=None)
    sd = scene_dir / 'session_1'

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        session_id = db.exec(select(Session).where(Session.scene_id == 'scene-13')).first().session_id

    (sd / 'metadata.json').unlink()

    with DBSession(engine) as db:
        with pytest.raises(MutationError):
            set_session_notes(db, session_id, 'hello')


def test_rename_task_cascades_to_every_member_scene(tmp_path: pathlib.Path):
    """Renaming a task rewrites scene.json for every scene assigned to it, and only those."""
    rec = tmp_path / 'recordings'
    _make_scene(rec, 'scene_2026-07-26_13-00-00_mnop', scene_id='scene-4', task='fold_towel')
    _make_scene(rec, 'scene_2026-07-26_14-00-00_qrst', scene_id='scene-5', task='fold_towel')
    other_dir = _make_scene(rec, 'scene_2026-07-26_15-00-00_uvwx', scene_id='scene-6', task='wipe_table')

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        task = db.exec(select(Task).where(Task.name == 'fold_towel')).first()
        rename_task(db, task.id, 'fold_towel_v2')

    for dirname in ('scene_2026-07-26_13-00-00_mnop', 'scene_2026-07-26_14-00-00_qrst'):
        manifest = SceneManifest.from_scene_dir(rec / dirname)
        assert manifest.task == 'fold_towel_v2'
    # the differently-tasked scene is untouched
    assert SceneManifest.from_scene_dir(other_dir).task == 'wipe_table'

    # dropped-and-resynced DB still reflects the rename
    rebuilt = get_engine(tmp_path / 'catalog2.db')
    sync_recordings(rec, rebuilt)
    with DBSession(rebuilt) as db:
        assert db.exec(select(Task).where(Task.name == 'fold_towel_v2')).first() is not None
        assert db.exec(select(Task).where(Task.name == 'fold_towel')).first() is None


def test_rename_task_rejects_name_collision(tmp_path: pathlib.Path):
    """Renaming a task to an existing different task's name is rejected."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        a = create_task(db, 'fold_towel')
        create_task(db, 'wipe_table')
        with pytest.raises(MutationError):
            rename_task(db, a.id, 'wipe_table')


def test_set_task_description_writes_db_row(tmp_path: pathlib.Path):
    """Setting a task's description updates the DB row (tasks have no on-disk manifest to write)."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        task = create_task(db, 'fold_towel')
        set_task_description(db, task.id, 'Fold the towel in half, then in half again.')
        assert db.get(Task, task.id).description == 'Fold the towel in half, then in half again.'


def test_set_task_description_blank_clears(tmp_path: pathlib.Path):
    """A blank/whitespace-only value clears the description rather than storing an empty string."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        task = create_task(db, 'fold_towel', description='old description')
        set_task_description(db, task.id, '   ')
        assert db.get(Task, task.id).description is None


def test_set_task_description_rejects_unknown_task(tmp_path: pathlib.Path):
    """An unknown task_id raises MutationError instead of a crash."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        with pytest.raises(MutationError):
            set_task_description(db, 999, 'hello')


def test_delete_scene_removes_directory_and_rows(tmp_path: pathlib.Path):
    """Deleting a scene removes its directory, its own row, and its session rows."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-26_10-00-00_abcd', scene_id='scene-1', task='fold_towel')
    keep_dir = _make_scene(rec, 'scene_2026-07-27_10-00-00_beef', scene_id='scene-2', task='fold_towel')

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        db.add(DatasetMember(dataset_id=1, scene_id='scene-1'))
        db.commit()
        delete_scene(db, 'scene-1', rec)

        assert not scene_dir.exists()
        assert keep_dir.exists()  # the neighbouring scene is untouched
        assert db.get(Scene, 'scene-1') is None
        assert db.exec(select(Session).where(Session.scene_id == 'scene-1')).all() == []
        assert db.exec(select(DatasetMember).where(DatasetMember.scene_id == 'scene-1')).all() == []
        assert db.get(Scene, 'scene-2') is not None


def test_delete_scene_refuses_directory_outside_recordings(tmp_path: pathlib.Path):
    """A row pointing outside the recordings tree is refused, and nothing is removed."""
    rec = tmp_path / 'recordings'
    _make_scene(rec, 'scene_2026-07-26_10-00-00_abcd', scene_id='scene-1', task=None)
    elsewhere = tmp_path / 'not_recordings'
    elsewhere.mkdir()

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        db.get(Scene, 'scene-1').dir = str(elsewhere)
        db.commit()
        with pytest.raises(MutationError, match='not inside'):
            delete_scene(db, 'scene-1', rec)
        assert elsewhere.exists()
        assert db.get(Scene, 'scene-1') is not None


def test_delete_scene_with_directory_already_gone_still_clears_rows(tmp_path: pathlib.Path):
    """A scene whose directory was removed outside the UI still deletes cleanly."""
    rec = tmp_path / 'recordings'
    scene_dir = _make_scene(rec, 'scene_2026-07-26_10-00-00_abcd', scene_id='scene-1', task=None)

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    shutil.rmtree(scene_dir)
    with DBSession(engine) as db:
        delete_scene(db, 'scene-1', rec)
        assert db.get(Scene, 'scene-1') is None

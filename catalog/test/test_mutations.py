"""
Tests for Phase 2 mutations: create/rename tasks and scene->task assignment.

The Phase 2 deliverable per docs/catalog-ui-plan.md §9 is: "retag a scene, drop the
DB, re-sync, tag survives" — i.e. every mutation here must write scene.json
(authoritative), not just the DB row. Each test below verifies that by dropping and
rebuilding the DB from a fresh engine and re-running sync.
"""

from __future__ import annotations

import pathlib

import pytest
from polyumi_catalog.db import get_engine
from polyumi_catalog.manifests import SceneManifest
from polyumi_catalog.models import Scene, Session, Task
from polyumi_catalog.mutations import (
    MutationError,
    assign_scene_task,
    create_task,
    rename_task,
    set_scene_notes,
    set_session_notes,
    set_session_unusable,
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

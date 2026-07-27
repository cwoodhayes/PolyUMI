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
from polyumi_catalog.models import Scene, Task
from polyumi_catalog.mutations import MutationError, assign_scene_task, create_task, rename_task
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

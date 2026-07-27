"""Tests for the recordings-tree sync scanner."""

from __future__ import annotations

import os
import pathlib
from datetime import datetime, timezone

from polyumi_catalog.db import get_engine
from polyumi_catalog.manifests import SceneManifest
from polyumi_catalog.models import Scene, Session, Task
from polyumi_catalog.sync import sync_recordings
from polyumi_pi.files.metadata import SessionMetadata, SessionType
from sqlmodel import Session as DBSession
from sqlmodel import select


def _make_session(scene_dir: pathlib.Path, name: str, *, scene_id: str, session_type: SessionType, task: str | None):
    """Create a session directory with a metadata.json under scene_dir."""
    sd = scene_dir / name
    sd.mkdir(parents=True)
    md = SessionMetadata(
        path=sd / 'metadata.json',
        scene_id=scene_id,
        session_type=session_type,
        task=task,
        n_video_frames=100,
    )
    md.to_file()
    return sd


def _engine(tmp_path: pathlib.Path):
    return get_engine(tmp_path / 'catalog.db')


def test_sync_populates_scenes_sessions_tasks(tmp_path: pathlib.Path):
    """A scene with a manifest + two sessions produces the expected rows."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-20_10-00-00_abcd'
    scene_dir.mkdir(parents=True)
    SceneManifest(scene_id='scene-1', task='fold_towel').write_to_scene_dir(scene_dir)
    _make_session(scene_dir, 'session_1', scene_id='scene-1', session_type=SessionType.MAPPING, task=None)
    _make_session(scene_dir, 'session_2', scene_id='scene-1', session_type=SessionType.EPISODE, task=None)

    engine = _engine(tmp_path)
    stats = sync_recordings(rec, engine)

    assert stats.scenes_updated == 1
    assert stats.sessions_upserted == 2
    assert stats.tasks_created == 1

    with DBSession(engine) as db:
        scene = db.get(Scene, 'scene-1')
        assert scene is not None
        task = db.get(Task, scene.task_id)
        assert task.name == 'fold_towel'
        sessions = db.exec(select(Session).where(Session.scene_id == 'scene-1')).all()
        assert {s.session_type for s in sessions} == {'MAPPING', 'EPISODE'}


def test_sync_scene_id_falls_back_to_metadata(tmp_path: pathlib.Path):
    """Without scene.json, the scene id comes from the sessions' metadata."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-21_10-00-00_efgh'
    scene_dir.mkdir(parents=True)
    _make_session(scene_dir, 'session_1', scene_id='meta-scene', session_type=SessionType.EPISODE, task=None)

    engine = _engine(tmp_path)
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        assert db.get(Scene, 'meta-scene') is not None


def test_sync_detects_task_conflict(tmp_path: pathlib.Path):
    """A session whose metadata task differs from scene.json is flagged, not silently merged."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-22_10-00-00_ijkl'
    scene_dir.mkdir(parents=True)
    SceneManifest(scene_id='scene-c', task='fold_towel').write_to_scene_dir(scene_dir)
    _make_session(scene_dir, 'session_1', scene_id='scene-c', session_type=SessionType.EPISODE, task='wipe_table')

    engine = _engine(tmp_path)
    stats = sync_recordings(rec, engine)
    assert len(stats.conflicts) == 1
    assert stats.conflicts[0].scene_task == 'fold_towel'
    assert stats.conflicts[0].meta_task == 'wipe_table'


def test_sync_is_idempotent_and_mtime_gated(tmp_path: pathlib.Path):
    """A second sync with no disk changes skips the scene."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-23_10-00-00_mnop'
    scene_dir.mkdir(parents=True)
    _make_session(scene_dir, 'session_1', scene_id='scene-d', session_type=SessionType.EPISODE, task=None)
    # ensure mtimes are safely in the past relative to the sync timestamp
    past = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    for p in (scene_dir, scene_dir / 'session_1', scene_dir / 'session_1' / 'metadata.json'):
        os.utime(p, (past, past))

    engine = _engine(tmp_path)
    sync_recordings(rec, engine)
    stats2 = sync_recordings(rec, engine)
    assert stats2.scenes_skipped == 1
    assert stats2.scenes_updated == 0


def test_sync_reconciles_removed_sessions(tmp_path: pathlib.Path):
    """Deleting a session dir removes its row on the next forced sync."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-24_10-00-00_qrst'
    scene_dir.mkdir(parents=True)
    _make_session(scene_dir, 'session_1', scene_id='scene-e', session_type=SessionType.EPISODE, task=None)
    sd2 = _make_session(scene_dir, 'session_2', scene_id='scene-e', session_type=SessionType.EPISODE, task=None)

    engine = _engine(tmp_path)
    sync_recordings(rec, engine)

    (sd2 / 'metadata.json').unlink()
    sd2.rmdir()
    stats = sync_recordings(rec, engine, force=True)
    assert stats.sessions_removed == 1
    with DBSession(engine) as db:
        assert len(db.exec(select(Session).where(Session.scene_id == 'scene-e')).all()) == 1


def test_sync_preserves_session_row_when_metadata_fails_to_parse(tmp_path: pathlib.Path):
    """A session row survives resync if its metadata.json merely fails to parse (not deleted)."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-26_10-00-00_yzab'
    scene_dir.mkdir(parents=True)
    # keep a second, healthy session in the scene so the scene's own identity (resolved from
    # metas[0] when there's no scene.json) doesn't shift once session_2's metadata is corrupted.
    _make_session(scene_dir, 'session_1', scene_id='scene-g', session_type=SessionType.EPISODE, task=None)
    sd2 = _make_session(scene_dir, 'session_2', scene_id='scene-g', session_type=SessionType.EPISODE, task=None)

    engine = _engine(tmp_path)
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        rows = db.exec(select(Session).where(Session.scene_id == 'scene-g')).all()
        assert len(rows) == 2
        session2_id = next(r.session_id for r in rows if r.dir == str(sd2))

    # corrupt session_2's metadata.json in place, then force a resync
    (sd2 / 'metadata.json').write_text('not valid json')
    stats = sync_recordings(rec, engine, force=True)

    assert stats.sessions_removed == 0
    with DBSession(engine) as db:
        row = db.get(Session, session2_id)
        assert row is not None
        assert row.dir == str(sd2)


def test_sync_retires_stale_scene_row_when_identity_recovers(tmp_path: pathlib.Path):
    """
    A scene resynced after unparseable metadata recovers doesn't leave a duplicate row.

    Without scene.json, an all-sessions-unparseable scene falls back to the directory name
    as its scene_id. Once metadata becomes parseable again, the real (metadata) scene_id
    resolves differently — the old fallback-id row must be retired, not left behind
    pointing at the same directory.
    """
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-26_16-00-00_yzcd'
    scene_dir.mkdir(parents=True)
    sd = _make_session(scene_dir, 'session_1', scene_id='scene-real-id', session_type=SessionType.EPISODE, task=None)

    # break metadata.json before the very first sync, so scene_id falls back to the dirname
    good_metadata = (sd / 'metadata.json').read_text()
    (sd / 'metadata.json').write_text('not valid json')

    engine = _engine(tmp_path)
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        fallback_scene = db.get(Scene, scene_dir.name)
        assert fallback_scene is not None

    # metadata recovers; resync should resolve the real scene_id and drop the fallback row
    (sd / 'metadata.json').write_text(good_metadata)
    sync_recordings(rec, engine, force=True)

    with DBSession(engine) as db:
        rows = db.exec(select(Scene).where(Scene.dir == str(scene_dir))).all()
        assert [s.scene_id for s in rows] == ['scene-real-id']


def test_sync_normalizes_whitespace_only_task_from_scene_json(tmp_path: pathlib.Path):
    """A scene.json task of only whitespace is treated as unassigned, not a blank Task row."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-26_17-00-00_yzef'
    scene_dir.mkdir(parents=True)
    SceneManifest(scene_id='scene-h', task='   ').write_to_scene_dir(scene_dir)
    _make_session(scene_dir, 'session_1', scene_id='scene-h', session_type=SessionType.EPISODE, task=None)

    engine = _engine(tmp_path)
    sync_recordings(rec, engine)

    with DBSession(engine) as db:
        scene = db.get(Scene, 'scene-h')
        assert scene.task_id is None
        assert db.exec(select(Task)).all() == []


def test_sync_flags_archived_scene(tmp_path: pathlib.Path):
    """A scene with a *.zarr.zip and no working scene.zarr is marked archived."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-25_10-00-00_uvwx'
    scene_dir.mkdir(parents=True)
    _make_session(scene_dir, 'session_1', scene_id='scene-f', session_type=SessionType.EPISODE, task=None)
    (scene_dir / 'scene.zarr.zip').write_bytes(b'PK\x03\x04')

    engine = _engine(tmp_path)
    sync_recordings(rec, engine)
    with DBSession(engine) as db:
        assert db.get(Scene, 'scene-f').archived is True

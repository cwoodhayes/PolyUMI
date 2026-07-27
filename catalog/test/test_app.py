"""HTTP-level tests for the catalog browser: Phase 1 reads plus Phase 2 mutations."""

from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient
from polyumi_catalog.app import create_app
from polyumi_catalog.db import get_engine
from polyumi_catalog.manifests import SceneManifest
from polyumi_catalog.sync import sync_recordings
from polyumi_pi.files.metadata import SessionMetadata, SessionType
from sqlmodel import Session as DBSession
from sqlmodel import select

from polyumi_catalog.models import Scene, Task


def _make_session(scene_dir: pathlib.Path, name: str, *, scene_id: str, session_type: SessionType):
    sd = scene_dir / name
    sd.mkdir(parents=True)
    md = SessionMetadata(path=sd / 'metadata.json', scene_id=scene_id, session_type=session_type, n_video_frames=10)
    md.to_file()
    return sd


def _seed(tmp_path: pathlib.Path):
    """Seed a recordings tree + synced engine, returning (recordings_dir, engine)."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    scene_dir.mkdir(parents=True)
    SceneManifest(scene_id='scene-1', task='fold_towel').write_to_scene_dir(scene_dir)
    _make_session(scene_dir, 'session_1', scene_id='scene-1', session_type=SessionType.EPISODE)

    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)
    return rec, engine


def _client(tmp_path: pathlib.Path, *, with_recordings: bool = True) -> TestClient:
    rec, engine = _seed(tmp_path)
    app = create_app(engine, recordings_dir=rec if with_recordings else None)
    return TestClient(app)


def test_index_lists_tasks(tmp_path: pathlib.Path):
    """The index page renders and includes the seeded task name."""
    resp = _client(tmp_path).get('/')
    assert resp.status_code == 200
    assert 'fold_towel' in resp.text


def test_select_task_returns_scenes_and_oob_detail(tmp_path: pathlib.Path):
    """Selecting a task returns its scenes plus out-of-band datasets/detail swaps."""
    client = _client(tmp_path)
    with client as c:
        tasks_resp = c.get('/')
    assert 'fold_towel' in tasks_resp.text

    # find the real task's id via the DB-backed page content isn't trivial from HTML,
    # so hit the query layer indirectly through 'all' which every scene matches.
    resp = client.get('/select/task/all')
    assert resp.status_code == 200
    assert 'scene_2026-07-26_10-00-00_abcd' in resp.text
    assert 'hx-swap-oob' in resp.text  # datasets + detail come back OOB


def test_select_scene_returns_sessions(tmp_path: pathlib.Path):
    """Selecting a scene returns its session list."""
    client = _client(tmp_path)
    resp = client.get('/select/scene/scene-1')
    assert resp.status_code == 200
    assert 'session_1' in resp.text
    assert 'EPISODE' in resp.text


def test_select_session_returns_detail(tmp_path: pathlib.Path):
    """Selecting a session returns its detail panel with type and directory."""
    client = _client(tmp_path)
    with client as c:
        # session_id is a uuid assigned by SessionMetadata; fetch it via the scene route
        scene_resp = c.get('/select/scene/scene-1')
    assert 'session_1' in scene_resp.text

    # Re-derive the session_id from disk since it's a generated uuid, not the dir name.
    from polyumi_pi.files.metadata import SessionMetadata as SM

    rec = tmp_path / 'recordings' / 'scene_2026-07-26_10-00-00_abcd' / 'session_1' / 'metadata.json'
    session_id = SM.from_file(rec).session_id

    resp = client.get(f'/select/session/{session_id}')
    assert resp.status_code == 200
    assert 'EPISODE' in resp.text


def test_unknown_scene_returns_empty_detail_not_error(tmp_path: pathlib.Path):
    """An unknown id renders the empty-state partial rather than a 404/500."""
    resp = _client(tmp_path).get('/select/scene/does-not-exist')
    assert resp.status_code == 200


def test_rescan_disabled_without_recordings_dir(tmp_path: pathlib.Path):
    """POST /rescan is a no-op (200, DB unchanged) when the app has no recordings_dir."""
    client = _client(tmp_path, with_recordings=False)
    resp = client.post('/rescan')
    assert resp.status_code == 200
    assert 'fold_towel' in resp.text


def test_no_get_route_mutates_state(tmp_path: pathlib.Path):
    """Sanity check: hitting every read route twice yields byte-identical responses."""
    client = _client(tmp_path)
    for path in ('/', '/select/task/all', '/select/scene/scene-1'):
        first = client.get(path).text
        second = client.get(path).text
        assert first == second


def test_select_scene_includes_assign_task_dropdown(tmp_path: pathlib.Path):
    """The scene detail panel offers a task dropdown populated with real tasks."""
    resp = _client(tmp_path).get('/select/scene/scene-1')
    assert 'name="task_id"' in resp.text
    assert '>fold_towel<' in resp.text


def test_post_create_task_redirects_and_persists(tmp_path: pathlib.Path):
    """POST /tasks creates a task and redirects to '/' without following it automatically."""
    rec, engine = _seed(tmp_path)
    app = create_app(engine, recordings_dir=rec)
    client = TestClient(app, follow_redirects=False)

    resp = client.post('/tasks', data={'name': 'wipe_table'})
    assert resp.status_code == 303
    assert resp.headers['location'] == '/'

    with DBSession(engine) as db:
        assert db.exec(select(Task).where(Task.name == 'wipe_table')).first() is not None


def test_post_create_task_rejects_blank_name(tmp_path: pathlib.Path):
    """A blank task name is rejected with 400, not silently accepted."""
    rec, engine = _seed(tmp_path)
    app = create_app(engine, recordings_dir=rec)
    client = TestClient(app, follow_redirects=False)

    resp = client.post('/tasks', data={'name': '   '})
    assert resp.status_code == 400


def test_post_assign_scene_task_writes_scene_json(tmp_path: pathlib.Path):
    """POST /scenes/{id}/task reassigns the scene and rewrites its scene.json."""
    rec, engine = _seed(tmp_path)
    with DBSession(engine) as db:
        wipe = Task(name='wipe_table')
        db.add(wipe)
        db.commit()
        wipe_id = wipe.id

    app = create_app(engine, recordings_dir=rec)
    client = TestClient(app, follow_redirects=False)
    resp = client.post('/scenes/scene-1/task', data={'task_id': str(wipe_id)})
    assert resp.status_code == 303

    with DBSession(engine) as db:
        assert db.get(Scene, 'scene-1').task_id == wipe_id
    manifest = SceneManifest.from_scene_dir(rec / 'scene_2026-07-26_10-00-00_abcd')
    assert manifest.task == 'wipe_table'


def test_post_assign_scene_task_unassign_with_empty_value(tmp_path: pathlib.Path):
    """Posting an empty task_id clears the scene's task (the '(unassigned)' option)."""
    rec, engine = _seed(tmp_path)
    app = create_app(engine, recordings_dir=rec)
    client = TestClient(app, follow_redirects=False)

    resp = client.post('/scenes/scene-1/task', data={'task_id': ''})
    assert resp.status_code == 303
    with DBSession(engine) as db:
        assert db.get(Scene, 'scene-1').task_id is None


def test_post_rename_task_cascades_and_redirects(tmp_path: pathlib.Path):
    """POST /tasks/{id}/rename updates the task and rewrites every member scene.json."""
    rec, engine = _seed(tmp_path)
    with DBSession(engine) as db:
        task = db.exec(select(Task).where(Task.name == 'fold_towel')).first()
        task_id = task.id

    app = create_app(engine, recordings_dir=rec)
    client = TestClient(app, follow_redirects=False)
    resp = client.post(f'/tasks/{task_id}/rename', data={'new_name': 'fold_towel_v2'})
    assert resp.status_code == 303

    manifest = SceneManifest.from_scene_dir(rec / 'scene_2026-07-26_10-00-00_abcd')
    assert manifest.task == 'fold_towel_v2'

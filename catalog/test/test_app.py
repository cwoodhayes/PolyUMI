"""HTTP-level tests for the catalog browser: Phase 1 reads plus Phase 2 mutations."""

from __future__ import annotations

import pathlib
import threading
import time

import zarr
from fastapi.testclient import TestClient
from polyumi_catalog.app import create_app
from polyumi_catalog.db import get_engine
from polyumi_catalog.manifests import SceneManifest
from polyumi_catalog.sync import sync_recordings, sync_scene_quality
from polyumi_pi.files.metadata import SessionMetadata, SessionType
from sqlmodel import Session as DBSession
from sqlmodel import select

from polyumi_catalog.models import Scene, Task
from polyumi_catalog.models import Session as SessionRow


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


def test_select_task_detail_includes_episode_count(tmp_path: pathlib.Path):
    """The task detail panel shows usable and total episode counts alongside the scene count."""
    resp = _client(tmp_path).get('/select/task/all')
    assert '<dt>Episodes</dt><dd>1 usable / 1 total</dd>' in resp.text


def test_scene_badge_counts_episodes_not_sessions(tmp_path: pathlib.Path):
    """The Scenes badge is usable-over-total *episodes*; mapping passes only show in the tooltip."""
    rec = tmp_path / 'recordings'
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    scene_dir.mkdir(parents=True)
    SceneManifest(scene_id='scene-1', task='fold_towel', unusable_episodes=['session_2']).write_to_scene_dir(scene_dir)
    _make_session(scene_dir, 'session_0', scene_id='scene-1', session_type=SessionType.MAPPING)
    _make_session(scene_dir, 'session_1', scene_id='scene-1', session_type=SessionType.EPISODE)
    _make_session(scene_dir, 'session_2', scene_id='scene-1', session_type=SessionType.EPISODE)
    engine = get_engine(tmp_path / 'catalog.db')
    sync_recordings(rec, engine)

    resp = TestClient(create_app(engine, recordings_dir=rec)).get('/select/task/all')
    # 1 usable of 2 episodes -- not 1/3, which would count the mapping pass against usability
    assert '>1/2</span>' in resp.text
    assert '1 usable of 2 episodes, 3 sessions in total' in resp.text


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


def test_select_session_shows_gopro_fps(tmp_path: pathlib.Path, monkeypatch):
    """
    The session detail panel shows the GoPro's native fps, read from its mp4 sidecar.

    The actual mp4 decoding is exercised in test_thumbnails.py; this only checks the
    query's glue (session -> its own directory -> thumbnails.gopro_fps -> template).
    """
    monkeypatch.setattr('polyumi_catalog.thumbnails.gopro_fps', lambda session_dir: 59.94)

    client = _client(tmp_path)
    from polyumi_pi.files.metadata import SessionMetadata as SM

    rec = tmp_path / 'recordings' / 'scene_2026-07-26_10-00-00_abcd' / 'session_1' / 'metadata.json'
    session_id = SM.from_file(rec).session_id

    resp = client.get(f'/select/session/{session_id}')
    assert '<dt>GoPro fps</dt><dd>59.9</dd>' in resp.text


def test_select_session_omits_gopro_fps_when_no_video(tmp_path: pathlib.Path):
    """A session with no gopro.mp4 sidecar (the seeded fixture has none) shows '—', not a crash."""
    from polyumi_pi.files.metadata import SessionMetadata as SM

    client = _client(tmp_path)
    rec = tmp_path / 'recordings' / 'scene_2026-07-26_10-00-00_abcd' / 'session_1' / 'metadata.json'
    session_id = SM.from_file(rec).session_id

    resp = client.get(f'/select/session/{session_id}')
    assert resp.status_code == 200
    assert '<dt>GoPro fps</dt><dd>—</dd>' in resp.text


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


def test_fetch_disabled_without_recordings_dir(tmp_path: pathlib.Path):
    """POST /fetch is rejected when the app has no recordings_dir to fetch into."""
    assert _client(tmp_path, with_recordings=False).post('/fetch').status_code == 400


def _fake_pi(monkeypatch, *, sessions: dict[str, list[str]], on_copy=None):
    """
    Patch app.PiFetch with a stub whose remote is ``{scene: [session, ...]}``.

    ``missing_sessions`` mirrors the real one — remote sessions minus what's already on disk —
    because that subtraction is exactly what lets a scene grow between fetches.
    """

    class FakePi:
        def __init__(self, host: str) -> None:
            self.host = host

        def list_remote_scenes(self) -> list[str]:
            return list(sessions)

        def missing_sessions(self, name: str, local_scene: pathlib.Path) -> list[str]:
            # Mirrors the real one, metadata.json completeness check included: a directory a
            # dropped transfer left behind is not a fetched session.
            local = (
                {
                    p.name
                    for p in local_scene.iterdir()
                    if p.is_dir() and p.name.startswith('session_') and (p / 'metadata.json').exists()
                }
                if local_scene.is_dir()
                else set()
            )
            return [s for s in sessions[name] if s not in local]

        def copy_sessions(
            self, name: str, session_names: list[str], local_parent: pathlib.Path, verbose: bool = False
        ) -> None:
            for session in session_names:
                on_copy(name, local_parent / name, session)

    monkeypatch.setattr('polyumi_catalog.app.PiFetch', FakePi)


def test_fetch_copies_only_new_sessions_then_syncs(tmp_path: pathlib.Path, monkeypatch):
    """/fetch skips sessions already local, copies the rest, and re-syncs so they appear in Tasks."""
    rec, engine = _seed(tmp_path)
    new_scene = 'scene_2026-07-27_11-00-00_beef'

    def copy(name: str, local_path: pathlib.Path, session: str) -> None:
        local_path.mkdir(parents=True, exist_ok=True)
        SceneManifest(scene_id='scene-2', task='wipe_table').write_to_scene_dir(local_path)
        _make_session(local_path, session, scene_id='scene-2', session_type=SessionType.EPISODE)

    _fake_pi(
        monkeypatch,
        sessions={'scene_2026-07-26_10-00-00_abcd': [], new_scene: ['session_1']},
        on_copy=copy,
    )
    app = create_app(engine, recordings_dir=rec, pi_host='fake')
    client = TestClient(app)

    assert client.post('/fetch').status_code == 200
    app.state.fetch_thread.join(timeout=30)
    assert app.state.fetch['status'] == 'done'
    assert app.state.fetch['total'] == 1  # the already-local scene contributed nothing
    assert (rec / new_scene).is_dir()

    poll = client.get('/fetch-poll').text
    assert 'fetched 1 session' in poll
    assert 'wipe_table' in poll  # OOB Tasks refresh, so the new scene is browsable
    assert 'hx-get="/fetch-poll"' not in poll  # polling stops once done


def test_fetch_picks_up_sessions_added_to_an_existing_scene(tmp_path: pathlib.Path, monkeypatch):
    """
    A scene already on disk is re-visited for the sessions it doesn't have yet.

    This is the mapping-then-episodes loop: the scene arrives with only its mapping walk, gets
    preprocessed, then grows as episodes are recorded into it on the Pi.
    """
    rec, engine = _seed(tmp_path)
    existing = 'scene_2026-07-26_10-00-00_abcd'
    copied: list[str] = []

    def copy(name: str, local_path: pathlib.Path, session: str) -> None:
        copied.append(session)
        _make_session(local_path, session, scene_id='scene-1', session_type=SessionType.EPISODE)

    local_sessions = [p.name for p in (rec / existing).iterdir() if p.name.startswith('session_')]
    _fake_pi(monkeypatch, sessions={existing: [*local_sessions, 'session_new']}, on_copy=copy)
    app = create_app(engine, recordings_dir=rec, pi_host='fake')
    client = TestClient(app)

    client.post('/fetch')
    app.state.fetch_thread.join(timeout=30)
    assert app.state.fetch['status'] == 'done'
    assert copied == ['session_new']  # only the new one, not the whole scene again


def test_fetch_failure_names_the_session_and_clears_only_its_dir(tmp_path: pathlib.Path, monkeypatch):
    """
    A half-finished transfer is reported, and only that session's directory is dropped.

    Removing the whole scene would take previously-fetched sessions, scene.zarr, scene.json and
    the SLAM atlas with it.
    """
    rec, engine = _seed(tmp_path)
    existing = 'scene_2026-07-26_10-00-00_abcd'
    kept = sorted(p.name for p in (rec / existing).iterdir() if p.name.startswith('session_'))

    def boom(name: str, local_path: pathlib.Path, session: str) -> None:
        (local_path / session / 'video').mkdir(parents=True)  # what tar leaves when the stream dies
        raise RuntimeError('ssh/tar sender failed with code 255')

    _fake_pi(monkeypatch, sessions={existing: [*kept, 'session_broken']}, on_copy=boom)
    app = create_app(engine, recordings_dir=rec, pi_host='fake')
    client = TestClient(app)

    client.post('/fetch')
    app.state.fetch_thread.join(timeout=30)
    assert app.state.fetch['status'] == 'error'
    status = client.get('/fetch-poll').text
    assert 'code 255' in status
    assert 'session_broken' in status  # which session failed, not just that one did
    # otherwise the missing-session filter would treat the stub as already fetched
    assert not (rec / existing / 'session_broken').exists()
    # and the rest of the scene is untouched
    assert (rec / existing).is_dir()
    assert sorted(p.name for p in (rec / existing).iterdir() if p.name.startswith('session_')) == kept


def test_delete_scene_removes_it_and_redirects(tmp_path: pathlib.Path):
    """The scene detail pane's Delete removes the directory and drops it from the browser."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec), follow_redirects=False)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'

    assert 'Delete scene' in client.get('/select/scene/scene-1').text
    resp = client.post('/scenes/scene-1/delete')
    assert resp.status_code == 303
    assert not scene_dir.exists()
    with DBSession(engine) as db:
        assert db.get(Scene, 'scene-1') is None


def test_delete_scene_button_hidden_without_recordings_dir(tmp_path: pathlib.Path):
    """No recordings dir means POST would 400, so the button isn't offered in the first place."""
    client = _client(tmp_path, with_recordings=False)
    assert 'Delete scene' not in client.get('/select/scene/scene-1').text
    assert client.post('/scenes/scene-1/delete').status_code == 400


def test_delete_scene_refused_while_pipeline_running(tmp_path: pathlib.Path):
    """A scene with a pipeline thread writing into it can't be deleted out from under it."""
    rec, engine = _seed(tmp_path)
    app = create_app(engine, recordings_dir=rec)
    app.state.pp_runs['scene-1'] = {'status': 'running', 'error': None}

    resp = TestClient(app).post('/scenes/scene-1/delete')
    assert resp.status_code == 400
    assert (rec / 'scene_2026-07-26_10-00-00_abcd').exists()


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


def test_post_task_description_saves_and_shows_in_textarea(tmp_path: pathlib.Path):
    """POST /tasks/{id}/description updates the task row and shows the new text on re-render."""
    rec, engine = _seed(tmp_path)
    with DBSession(engine) as db:
        task_id = db.exec(select(Task).where(Task.name == 'fold_towel')).first().id

    client = TestClient(create_app(engine, recordings_dir=rec))
    resp = client.post(f'/tasks/{task_id}/description', data={'description': 'Fold in half, then in half again.'})
    assert resp.status_code == 200
    assert 'Fold in half, then in half again.</textarea>' in resp.text

    with DBSession(engine) as db:
        assert db.get(Task, task_id).description == 'Fold in half, then in half again.'


def test_post_task_description_unknown_task_returns_400(tmp_path: pathlib.Path):
    """Saving a description for a nonexistent task is rejected, not a crash."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).post('/tasks/999/description', data={'description': 'hi'})
    assert resp.status_code == 400


def _session_id(rec: pathlib.Path) -> str:
    from polyumi_pi.files.metadata import SessionMetadata as SM

    md = rec / 'scene_2026-07-26_10-00-00_abcd' / 'session_1' / 'metadata.json'
    return SM.from_file(md).session_id


def test_select_scene_shows_pzarr_hint_when_absent(tmp_path: pathlib.Path):
    """Without a scene.zarr, the session detail explains what's needed instead of showing buttons."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).get(f'/select/session/{_session_id(rec)}')
    assert 'Build pzarr first' in resp.text
    assert 'Export to MCAP' not in resp.text


def test_export_mcap_unknown_session_returns_404(tmp_path: pathlib.Path):
    """Exporting a nonexistent session id is a clean 404, not a crash."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).post('/sessions/does-not-exist/export-mcap')
    assert resp.status_code == 404


def test_export_mcap_without_pzarr_returns_400(tmp_path: pathlib.Path):
    """Exporting before pzarr exists is rejected with a clear message, not a 500."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).post(f'/sessions/{_session_id(rec)}/export-mcap')
    assert resp.status_code == 400
    assert 'pzarr' in resp.text.lower()


def test_export_mcap_success_then_open_foxglove(tmp_path: pathlib.Path, monkeypatch):
    """A successful export flips the detail panel to offer 'Open in Foxglove'; that route launches it."""
    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 1
    ep = root.require_group('episode_0')
    ep.attrs['session_dir'] = 'session_1'

    def fake_export_scene_to_mcap(scene_path, episode=None, **kwargs):
        out = scene_path / f'episode_{episode}.mcap'
        out.write_bytes(b'fake')
        return [out]

    monkeypatch.setattr('polyumi_ingest.export.mcap.export_scene_to_mcap', fake_export_scene_to_mcap)

    client = TestClient(create_app(engine, recordings_dir=rec))
    session_id = _session_id(rec)

    export_resp = client.post(f'/sessions/{session_id}/export-mcap')
    assert export_resp.status_code == 200
    assert 'Re-export MCAP' in export_resp.text
    assert 'Open in Foxglove' in export_resp.text
    assert (scene_dir / 'episode_0.mcap').is_file()

    launches = []
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/foxglove-studio')
    monkeypatch.setattr('subprocess.Popen', lambda args, **kwargs: launches.append(args))

    open_resp = client.post(f'/sessions/{session_id}/open-foxglove')
    assert open_resp.status_code == 204
    assert launches == [['/usr/bin/foxglove-studio', str(scene_dir / 'episode_0.mcap')]]


def test_open_foxglove_without_mcap_returns_400(tmp_path: pathlib.Path):
    """Opening Foxglove before any MCAP has been exported is rejected, not silently a no-op."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).post(f'/sessions/{_session_id(rec)}/open-foxglove')
    assert resp.status_code == 400


def test_post_scene_notes_saves_and_shows_in_textarea(tmp_path: pathlib.Path):
    """POST /scenes/{id}/notes rewrites scene.json and shows the new text on re-render."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))

    resp = client.post('/scenes/scene-1/notes', data={'notes': 'left-handed grasp'})
    assert resp.status_code == 200
    assert 'left-handed grasp</textarea>' in resp.text

    manifest = SceneManifest.from_scene_dir(rec / 'scene_2026-07-26_10-00-00_abcd')
    assert manifest.notes == 'left-handed grasp'


def test_post_scene_notes_unknown_scene_returns_400(tmp_path: pathlib.Path):
    """Saving notes for a nonexistent scene is rejected, not a crash."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).post('/scenes/no-such-scene/notes', data={'notes': 'hi'})
    assert resp.status_code == 400


def test_post_session_notes_saves_and_shows_in_textarea(tmp_path: pathlib.Path):
    """POST /sessions/{id}/notes rewrites metadata.json and shows the new text on re-render."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))
    session_id = _session_id(rec)

    resp = client.post(f'/sessions/{session_id}/notes', data={'notes': 'gripper slipped'})
    assert resp.status_code == 200
    assert 'gripper slipped</textarea>' in resp.text

    md_path = rec / 'scene_2026-07-26_10-00-00_abcd' / 'session_1' / 'metadata.json'
    from polyumi_pi.files.metadata import SessionMetadata as SM

    assert SM.from_file(md_path).notes == 'gripper slipped'


def test_post_session_notes_unknown_session_returns_400(tmp_path: pathlib.Path):
    """Saving notes for a nonexistent session is rejected, not a crash."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).post(
        '/sessions/does-not-exist/notes', data={'notes': 'hi'}
    )
    assert resp.status_code == 400


def test_mark_unusable_then_usable_round_trip(tmp_path: pathlib.Path):
    """Marking a session unusable then usable toggles the badge, scene.json, and the DB row."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))
    session_id = _session_id(rec)

    resp = client.post(f'/sessions/{session_id}/mark-unusable')
    assert resp.status_code == 200
    assert 'Mark usable' in resp.text
    assert 'unusable — excluded from exports' in resp.text
    # OOB episodes-column fragment is included in the same response
    assert 'item-unusable' in resp.text
    with DBSession(engine) as db:
        assert db.get(Scene, 'scene-1')  # sanity: scene still resolves
    manifest = SceneManifest.from_scene_dir(rec / 'scene_2026-07-26_10-00-00_abcd')
    assert manifest.unusable_episodes == ['session_1']

    resp = client.post(f'/sessions/{session_id}/mark-usable')
    assert resp.status_code == 200
    assert 'Mark unusable' in resp.text
    assert 'item-unusable' not in resp.text
    manifest = SceneManifest.from_scene_dir(rec / 'scene_2026-07-26_10-00-00_abcd')
    assert manifest.unusable_episodes == []


def test_mark_unusable_unknown_session_returns_400(tmp_path: pathlib.Path):
    """Marking a nonexistent session id is rejected, not a crash."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).post('/sessions/does-not-exist/mark-unusable')
    assert resp.status_code == 400


def test_pose_source_round_trip(tmp_path: pathlib.Path):
    """Setting a pose-source override then clearing it back to default updates the selector + scene.json."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))
    session_id = _session_id(rec)

    resp = client.post(f'/sessions/{session_id}/pose-source', data={'source': 'slam'})
    assert resp.status_code == 200
    manifest = SceneManifest.from_scene_dir(rec / 'scene_2026-07-26_10-00-00_abcd')
    assert manifest.pose_source_overrides == {'session_1': 'slam'}

    resp = client.post(f'/sessions/{session_id}/pose-source', data={'source': 'default'})
    assert resp.status_code == 200
    manifest = SceneManifest.from_scene_dir(rec / 'scene_2026-07-26_10-00-00_abcd')
    assert manifest.pose_source_overrides == {}


def test_pose_source_unknown_session_returns_400(tmp_path: pathlib.Path):
    """Setting a pose source on a nonexistent session id is rejected, not a crash."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).post(
        '/sessions/does-not-exist/pose-source', data={'source': 'slam'}
    )
    assert resp.status_code == 400


def test_pose_source_unknown_value_returns_400(tmp_path: pathlib.Path):
    """An unrecognized source value is rejected rather than silently accepted."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))
    resp = client.post(f'/sessions/{_session_id(rec)}/pose-source', data={'source': 'lidar'})
    assert resp.status_code == 400


def test_index_includes_empty_dataset_builder(tmp_path: pathlib.Path):
    """With no scenes added yet, the builder shows its empty state, still offering a task picker."""
    resp = _client(tmp_path).get('/')
    assert 'New Dataset Builder' in resp.text
    assert 'name="scene_ids"' not in resp.text
    assert 'name="task_id"' in resp.text
    assert '>fold_towel<' in resp.text


def test_index_omits_dataset_builder_without_recordings_dir(tmp_path: pathlib.Path):
    """Without a recordings_dir there's nowhere to export to, so the builder is hidden."""
    resp = _client(tmp_path, with_recordings=False).get('/')
    assert 'New Dataset Builder' not in resp.text


def test_post_dataset_draft_add_appears_on_index(tmp_path: pathlib.Path):
    """Adding a scene from its detail pane makes it show up in the builder on the next page load."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))

    add_resp = client.post('/dataset-draft/add/scene-1')
    assert add_resp.status_code == 200
    assert 'hx-swap-oob' in add_resp.text
    assert 'scene_2026-07-26_10-00-00_abcd' in add_resp.text

    index_resp = client.get('/')
    assert 'name="scene_ids" value="scene-1"' in index_resp.text
    assert index_resp.text.count('<li>') == 1


def test_post_dataset_draft_add_is_idempotent(tmp_path: pathlib.Path):
    """Adding the same scene twice doesn't duplicate it in the builder."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))

    client.post('/dataset-draft/add/scene-1')
    client.post('/dataset-draft/add/scene-1')

    resp = client.get('/')
    assert resp.text.count('name="scene_ids"') == 1


def test_post_dataset_draft_add_lock_prevents_concurrent_duplicates(tmp_path: pathlib.Path):
    """
    Many near-simultaneous 'add' POSTs for the same scene still only add it once.

    Regression test for the check-then-append race on app.state.pending_dataset_scene_ids
    (same class of bug as the pp-run race): without the lock, threads released at the same
    instant could all observe 'not yet in the list' before any of them appended.
    """
    rec, engine = _seed(tmp_path)
    n_threads = 8
    app = create_app(engine, recordings_dir=rec)
    barrier = threading.Barrier(n_threads)

    def _post():
        barrier.wait(timeout=5)
        TestClient(app).post('/dataset-draft/add/scene-1')

    threads = [threading.Thread(target=_post) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert app.state.pending_dataset_scene_ids == ['scene-1']


def test_post_dataset_draft_add_unknown_scene_returns_404(tmp_path: pathlib.Path):
    """Adding a nonexistent scene id is a clean 404, not a crash."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))
    resp = client.post('/dataset-draft/add/no-such-scene')
    assert resp.status_code == 404


def test_post_dataset_draft_remove(tmp_path: pathlib.Path):
    """Removing a scene takes it back out of the builder."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))

    client.post('/dataset-draft/add/scene-1')
    remove_resp = client.post('/dataset-draft/remove/scene-1')
    assert remove_resp.status_code == 200

    resp = client.get('/')
    assert 'name="scene_ids"' not in resp.text


def test_post_dataset_draft_remove_never_added_is_a_noop(tmp_path: pathlib.Path):
    """Removing a scene that was never added doesn't error."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))
    resp = client.post('/dataset-draft/remove/scene-1')
    assert resp.status_code == 200


def test_post_dataset_draft_from_dataset_seeds_builder_with_its_scenes(tmp_path: pathlib.Path, monkeypatch):
    """'Duplicate into builder' loads an existing dataset's member scenes into the draft, replacing it."""

    def fake_export_scenes_to_dp(scene_paths, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b'fake-zip')
        return 3, []

    monkeypatch.setattr('polyumi_ingest.export.dp.export_scenes_to_dp', fake_export_scenes_to_dp)

    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))
    client.post('/dataset-draft/add/scene-1')
    client.post('/datasets', data={'name': 'fold_towel_v1', 'scene_ids': ['scene-1']})
    with DBSession(engine) as db:
        from polyumi_catalog.models import Dataset

        dataset_id = db.exec(select(Dataset).where(Dataset.name == 'fold_towel_v1')).first().id

    # a leftover draft entry should be replaced, not appended to
    client.post('/dataset-draft/add/scene-1')
    resp = client.post(f'/dataset-draft/from-dataset/{dataset_id}')
    assert resp.status_code == 200
    assert 'hx-swap-oob' in resp.text

    index_resp = client.get('/')
    assert index_resp.text.count('name="scene_ids" value="scene-1"') == 1


def test_post_dataset_draft_from_dataset_unknown_id_returns_404(tmp_path: pathlib.Path):
    """Duplicating a nonexistent dataset id is a clean 404, not a crash."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))
    resp = client.post('/dataset-draft/from-dataset/999')
    assert resp.status_code == 404


def test_post_build_dataset_success_redirects_and_clears_draft(tmp_path: pathlib.Path, monkeypatch):
    """Building a dataset (via the draft-populated hidden inputs) redirects and clears the draft."""

    def fake_export_scenes_to_dp(scene_paths, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b'fake-zip')
        return 3, []

    monkeypatch.setattr('polyumi_ingest.export.dp.export_scenes_to_dp', fake_export_scenes_to_dp)

    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec), follow_redirects=False)
    client.post('/dataset-draft/add/scene-1')

    resp = client.post('/datasets', data={'name': 'fold_towel_v1', 'scene_ids': ['scene-1']})

    assert resp.status_code == 303
    assert resp.headers['location'] == '/'
    with DBSession(engine) as db:
        from polyumi_catalog.models import Dataset

        dataset = db.exec(select(Dataset).where(Dataset.name == 'fold_towel_v1')).first()
        assert dataset is not None
        assert dataset.n_episodes == 3
        assert dataset.task_id is None
    assert (rec / 'datasets' / 'fold_towel_v1.zarr.zip').is_file()
    assert (rec / 'datasets' / 'fold_towel_v1.dataset.json').is_file()

    # the draft is cleared after a successful build, and the new dataset is on the page the
    # POST redirects to — the Datasets column is otherwise only filled by /select/task
    index_resp = client.get('/')
    assert 'name="scene_ids"' not in index_resp.text
    assert 'fold_towel_v1' in index_resp.text
    assert 'No datasets.' not in index_resp.text


def test_post_build_dataset_with_task_id_persists_it(tmp_path: pathlib.Path, monkeypatch):
    """Selecting a task in the builder form actually associates the built dataset with it."""

    def fake_export_scenes_to_dp(scene_paths, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b'fake-zip')
        return 1, []

    monkeypatch.setattr('polyumi_ingest.export.dp.export_scenes_to_dp', fake_export_scenes_to_dp)

    rec, engine = _seed(tmp_path)
    with DBSession(engine) as db:
        task = db.exec(select(Task).where(Task.name == 'fold_towel')).first()
        task_id = task.id

    client = TestClient(create_app(engine, recordings_dir=rec), follow_redirects=False)
    resp = client.post('/datasets', data={'name': 'fold_towel_v1', 'task_id': str(task_id), 'scene_ids': ['scene-1']})
    assert resp.status_code == 303

    with DBSession(engine) as db:
        from polyumi_catalog.models import Dataset

        dataset = db.exec(select(Dataset).where(Dataset.name == 'fold_towel_v1')).first()
        assert dataset.task_id == task_id

    # and it now shows up when the Datasets column is filtered to that task
    task_resp = client.get(f'/select/task/{task_id}')
    assert 'fold_towel_v1' in task_resp.text


def test_post_build_dataset_without_recordings_dir_returns_400(tmp_path: pathlib.Path):
    """Building a dataset with no recordings_dir configured is rejected, not a crash."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=None), follow_redirects=False)
    resp = client.post('/datasets', data={'name': 'x', 'scene_ids': ['scene-1']})
    assert resp.status_code == 400


def test_post_build_dataset_rejects_no_scenes_selected(tmp_path: pathlib.Path):
    """Submitting the form with no scenes added is a clean 400, not an empty dataset."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec), follow_redirects=False)
    resp = client.post('/datasets', data={'name': 'x'})
    assert resp.status_code == 400


def test_get_thumbnail_404_for_unknown_session(tmp_path: pathlib.Path):
    """An unknown session_id 404s rather than crashing."""
    resp = _client(tmp_path).get('/sessions/does-not-exist/thumbnail.jpg')
    assert resp.status_code == 404


def test_get_thumbnail_404_when_no_gopro_mp4(tmp_path: pathlib.Path):
    """A real session with no gopro.mp4 sidecar (the seeded fixture has none) 404s cleanly."""
    from polyumi_pi.files.metadata import SessionMetadata as SM

    client = _client(tmp_path)
    rec = tmp_path / 'recordings' / 'scene_2026-07-26_10-00-00_abcd' / 'session_1' / 'metadata.json'
    session_id = SM.from_file(rec).session_id

    resp = client.get(f'/sessions/{session_id}/thumbnail.jpg')
    assert resp.status_code == 404


def test_get_thumbnail_returns_jpeg_when_decodable(tmp_path: pathlib.Path, monkeypatch):
    """
    The route serves the bytes thumbnails.session_thumbnail_jpeg produces, with a long cache header.

    The actual mp4 decoding is exercised in test_thumbnails.py; this only checks the route's glue
    (session lookup -> path resolution -> response headers).
    """
    from polyumi_pi.files.metadata import SessionMetadata as SM

    client = _client(tmp_path)
    rec = tmp_path / 'recordings' / 'scene_2026-07-26_10-00-00_abcd' / 'session_1' / 'metadata.json'
    session_id = SM.from_file(rec).session_id

    fake_jpeg = b'\xff\xd8fake-jpeg-bytes'
    monkeypatch.setattr('polyumi_catalog.thumbnails.session_thumbnail_jpeg', lambda session_dir: fake_jpeg)

    resp = client.get(f'/sessions/{session_id}/thumbnail.jpg')
    assert resp.status_code == 200
    assert resp.content == fake_jpeg
    assert resp.headers['content-type'] == 'image/jpeg'
    assert 'immutable' in resp.headers['cache-control']


def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    """Poll predicate() until truthy, or raise once timeout elapses (background-thread tests)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('condition never became true within timeout')


def test_run_pp_unknown_scene_returns_404(tmp_path: pathlib.Path):
    """Triggering the pipeline for a nonexistent scene is a clean 404, not a crash."""
    rec, engine = _seed(tmp_path)
    resp = TestClient(create_app(engine, recordings_dir=rec)).post('/scenes/does-not-exist/run-pp')
    assert resp.status_code == 404


def test_run_pp_shows_running_then_done(tmp_path: pathlib.Path, monkeypatch):
    """POST /run-pp returns immediately with a running indicator; it clears once the thread finishes."""
    rec, engine = _seed(tmp_path)
    started = threading.Event()
    finish = threading.Event()

    def fake_run_full_pipeline(scene_dir, force=False):
        started.set()
        finish.wait(timeout=5)

    monkeypatch.setattr('polyumi_catalog.pp_status.run_full_pipeline', fake_run_full_pipeline)

    client = TestClient(create_app(engine, recordings_dir=rec))
    resp = client.post('/scenes/scene-1/run-pp')
    assert resp.status_code == 200
    assert 'Running pipeline' in resp.text
    assert started.wait(timeout=5)

    poll_resp = client.get('/select/scene/scene-1')
    assert 'Running pipeline' in poll_resp.text
    assert 'Run full pipeline' not in poll_resp.text

    finish.set()
    _wait_until(lambda: 'Run full pipeline' in client.get('/select/scene/scene-1').text)


def test_run_pp_records_error_on_failure(tmp_path: pathlib.Path, monkeypatch):
    """A failing pipeline run surfaces its error message instead of leaving status stuck."""
    rec, engine = _seed(tmp_path)

    def failing(scene_dir, force=False):
        raise RuntimeError('missing gopro.mp4 in session_1')

    monkeypatch.setattr('polyumi_catalog.pp_status.run_full_pipeline', failing)

    client = TestClient(create_app(engine, recordings_dir=rec))
    client.post('/scenes/scene-1/run-pp')

    _wait_until(lambda: 'Last run failed' in client.get('/select/scene/scene-1').text)
    assert 'missing gopro.mp4' in client.get('/select/scene/scene-1').text


def test_run_pp_is_idempotent_while_already_running(tmp_path: pathlib.Path, monkeypatch):
    """A second POST while a run is in flight doesn't start a second background run."""
    rec, engine = _seed(tmp_path)
    calls = []
    started = threading.Event()
    finish = threading.Event()

    def fake_run_full_pipeline(scene_dir, force=False):
        calls.append(scene_dir)
        started.set()
        finish.wait(timeout=5)

    monkeypatch.setattr('polyumi_catalog.pp_status.run_full_pipeline', fake_run_full_pipeline)

    client = TestClient(create_app(engine, recordings_dir=rec))
    client.post('/scenes/scene-1/run-pp')
    assert started.wait(timeout=5)
    client.post('/scenes/scene-1/run-pp')
    finish.set()
    _wait_until(lambda: 'Run full pipeline' in client.get('/select/scene/scene-1').text)

    assert len(calls) == 1


def test_select_scene_includes_pp_status(tmp_path: pathlib.Path):
    """The scene detail panel shows pipeline step names even with no pzarr built yet."""
    resp = _client(tmp_path).get('/select/scene/scene-1')
    assert 'Preprocessing pipeline' in resp.text
    assert 'No pzarr built yet' in resp.text
    assert 'Run full pipeline' in resp.text


def test_run_pp_button_disables_itself_and_has_no_confirm_when_not_fully_done(tmp_path: pathlib.Path):
    """The button self-disables on click (hx-disabled-elt) and doesn't prompt when nothing's complete yet."""
    resp = _client(tmp_path).get('/select/scene/scene-1')
    assert 'hx-disabled-elt="this"' in resp.text
    assert 'hx-confirm' not in resp.text


def test_run_pp_button_confirms_when_scene_already_fully_processed(tmp_path: pathlib.Path):
    """Once every registered step is already complete, re-running asks for confirmation first."""
    from polyumi_ingest.preproc import available_preprocessing_steps

    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [s.step_number for s in available_preprocessing_steps()]

    resp = TestClient(create_app(engine, recordings_dir=rec)).get('/select/scene/scene-1')
    assert 'hx-confirm=' in resp.text
    assert 'already completed all' in resp.text
    # "Run full pipeline" would be a no-op here (nothing left to continue) — hidden.
    assert 'Run full pipeline' not in resp.text
    assert 'Re-run pipeline' in resp.text


def test_run_pp_no_rerun_button_when_nothing_complete_yet(tmp_path: pathlib.Path):
    """With zero steps done, there's nothing to force-redo, so only the plain button shows."""
    resp = _client(tmp_path).get('/select/scene/scene-1')
    assert 'Run full pipeline' in resp.text
    assert 'Re-run pipeline' not in resp.text


def test_run_pp_both_buttons_and_no_confirm_wording_when_partially_complete(tmp_path: pathlib.Path):
    """Some-but-not-all steps done: both buttons show; re-run confirms without the 'all N' wording."""
    from polyumi_ingest.preproc import available_preprocessing_steps

    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    all_steps = [s.step_number for s in available_preprocessing_steps()]
    root.attrs['preprocessing_steps'] = all_steps[:1]  # partial completion

    resp = TestClient(create_app(engine, recordings_dir=rec)).get('/select/scene/scene-1')
    assert 'Run full pipeline' in resp.text
    assert 'Re-run pipeline' in resp.text
    assert 'already completed all' not in resp.text
    assert 'discard the 1 already-completed step' in resp.text


def test_run_pp_force_true_is_passed_through_to_run_full_pipeline(tmp_path: pathlib.Path, monkeypatch):
    """Posting force=true reaches pp_status.run_full_pipeline as force=True, not silently dropped."""
    rec, engine = _seed(tmp_path)
    calls = []

    def fake_run_full_pipeline(scene_dir, force=False):
        calls.append(force)

    monkeypatch.setattr('polyumi_catalog.pp_status.run_full_pipeline', fake_run_full_pipeline)

    client = TestClient(create_app(engine, recordings_dir=rec))
    client.post('/scenes/scene-1/run-pp', data={'force': 'true'})
    _wait_until(lambda: len(calls) == 1)

    assert calls == [True]


def test_run_pp_omitted_force_defaults_to_false(tmp_path: pathlib.Path, monkeypatch):
    """The plain 'continue' button (no force field posted) must not force a redo."""
    rec, engine = _seed(tmp_path)
    calls = []

    def fake_run_full_pipeline(scene_dir, force=False):
        calls.append(force)

    monkeypatch.setattr('polyumi_catalog.pp_status.run_full_pipeline', fake_run_full_pipeline)

    client = TestClient(create_app(engine, recordings_dir=rec))
    client.post('/scenes/scene-1/run-pp')
    _wait_until(lambda: len(calls) == 1)

    assert calls == [False]


def test_run_pp_lock_prevents_concurrent_duplicate_runs(tmp_path: pathlib.Path, monkeypatch):
    """
    Many near-simultaneous POSTs to run-pp start the pipeline exactly once.

    Regression test for the check-then-act race on app.state.pp_runs: without the lock
    around the check-and-set, threads released by the barrier at the same instant could
    all observe 'not running' before any of them wrote 'running', each starting its own
    background pipeline run against the same scene.zarr.
    """
    rec, engine = _seed(tmp_path)
    n_threads = 8
    calls: list[pathlib.Path] = []
    calls_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)
    finish = threading.Event()

    def fake_run_full_pipeline(scene_dir, force=False):
        with calls_lock:
            calls.append(scene_dir)
        finish.wait(timeout=5)

    monkeypatch.setattr('polyumi_catalog.pp_status.run_full_pipeline', fake_run_full_pipeline)

    app = create_app(engine, recordings_dir=rec)

    def _post():
        barrier.wait(timeout=5)
        TestClient(app).post('/scenes/scene-1/run-pp')

    threads = [threading.Thread(target=_post) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    finish.set()

    assert len(calls) == 1


def test_scene_detail_shows_recording_and_pzarr_commits(tmp_path: pathlib.Path):
    """The scene pane surfaces both lineages: what recorded the sessions and what built the pzarr."""
    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['git_sha'] = 'c' * 40
    root.attrs['pipeline_version'] = '0.1.0'

    resp = TestClient(create_app(engine, recordings_dir=rec)).get('/select/scene/scene-1')

    assert 'Recorded by' in resp.text
    assert 'pzarr built by' in resp.text
    # abbreviated in the text, full sha available on hover
    assert 'c' * 12 in resp.text
    assert 'c' * 40 in resp.text
    assert '0.1.0' in resp.text


def test_scene_detail_shows_per_step_commit_for_completed_steps(tmp_path: pathlib.Path):
    """Each completed step renders the commit it was run under, so a mixed corpus is visible."""
    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [2]
    root.attrs['preprocessing_step_versions'] = {
        '2': {'git_sha': 'd' * 40, 'completed_at': '2026-08-01T00:00:00+00:00'},
    }

    resp = TestClient(create_app(engine, recordings_dir=rec)).get('/select/scene/scene-1')

    assert 'd' * 12 in resp.text
    assert '2026-08-01T00:00:00+00:00' in resp.text


def test_scene_detail_renders_without_provenance_attrs(tmp_path: pathlib.Path):
    """A store predating provenance still renders; the fields fall back to a dash."""
    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [2]

    resp = TestClient(create_app(engine, recordings_dir=rec)).get('/select/scene/scene-1')

    assert resp.status_code == 200
    assert 'Provenance' in resp.text


def test_run_pp_force_resets_steps_before_the_response_is_rendered(tmp_path: pathlib.Path, monkeypatch):
    """
    Clicking "Re-run pipeline" immediately shows 0/N, without waiting for the first poll.

    The reset is what makes it obvious the scene is being worked on; if it only happened
    on the background thread, this response could still claim every step was complete.
    """
    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [1, 2]

    started = threading.Event()
    release = threading.Event()

    def fake_run_full_pipeline(scene_dir, force=False):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr('polyumi_catalog.pp_status.run_full_pipeline', fake_run_full_pipeline)

    from polyumi_ingest.preproc import available_preprocessing_steps

    n_steps = len(available_preprocessing_steps())
    client = TestClient(create_app(engine, recordings_dir=rec))
    try:
        resp = client.post('/scenes/scene-1/run-pp', data={'force': 'true'})
        assert zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r').attrs['preprocessing_steps'] == []
        assert f'0 / {n_steps}' in resp.text
        assert 'pp-step-done' not in resp.text
        assert 'Running pipeline' in resp.text
    finally:
        release.set()
        started.wait(timeout=5)


def test_run_pp_without_force_leaves_completed_steps_ticked(tmp_path: pathlib.Path, monkeypatch):
    """The "continue" run must not reset — it exists precisely to skip what's already done."""
    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [1, 2]

    monkeypatch.setattr('polyumi_catalog.pp_status.run_full_pipeline', lambda *a, **k: None)

    client = TestClient(create_app(engine, recordings_dir=rec))
    client.post('/scenes/scene-1/run-pp')

    assert zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r').attrs['preprocessing_steps'] == [1, 2]


def test_pp_poll_reads_the_db_and_never_the_store(tmp_path: pathlib.Path):
    """
    The poll must stay a cheap render: it fires every 3s for the whole run.

    Measurements reach it via the run's own sync_scene_quality when the pipeline finishes,
    which the next tick then swaps in — not by the poll re-reading every episode's zarr attrs
    on every tick. Guards against putting that back.
    """
    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 1
    ep = root.require_group('episode_0')
    ep.attrs['session_dir'] = 'session_1'
    slam = ep.require_group('annotations').require_group('slam')
    slam.attrs['n_frames_total'] = 100
    slam.attrs['n_frames_lost'] = 90
    slam.attrs['tracking_ratio'] = 0.1
    slam.attrs['n_relocalization_events'] = 0

    app = create_app(engine, recordings_dir=rec)
    app.state.pp_runs['scene-1'] = {'status': 'running', 'error': None}
    client = TestClient(app)

    # Mid-run: the attrs are on disk but nothing has synced them, and the poll must not.
    resp = client.get('/scenes/scene-1/pp-poll')
    assert resp.status_code == 200
    assert 'col-episodes-body' in resp.text and 'hx-swap-oob' in resp.text
    assert '10% tracked' not in resp.text
    with DBSession(engine) as db:
        row = db.exec(select(SessionRow).where(SessionRow.dir.like('%session_1'))).one()
        assert row.slam_attrs_json is None

    # Run completion syncs once; the next tick is what puts the badge on screen.
    sync_scene_quality('scene-1', engine)
    assert '10% tracked' in client.get('/scenes/scene-1/pp-poll').text


def test_pp_poll_does_not_replace_the_notes_box(tmp_path: pathlib.Path):
    """
    The poll swaps the pipeline panel, not the whole pane, so mid-run typing survives.

    It fires every 3s for the length of a run. Swapping #detail-body replaced the Notes
    textarea along with everything else, so anything typed into it during a SLAM run was gone
    within three seconds — with no error and nothing to suggest where it went.
    """
    rec, engine = _seed(tmp_path)
    app = create_app(engine, recordings_dir=rec)
    app.state.pp_runs['scene-1'] = {'status': 'running', 'error': None}

    body = TestClient(app).get('/scenes/scene-1/pp-poll').text

    # the pipeline panel is the swap target, and the poll re-arms itself from inside it
    assert 'id="scene-pp-panel"' in body
    assert 'hx-target="#scene-pp-panel"' in body
    # nothing that carries user input comes back, so nothing of theirs is overwritten
    assert 'scene-notes' not in body
    assert 'assign-task-form' not in body
    assert 'id="detail-body"' not in body


def test_rescan_refreshes_the_open_scene_pane_with_new_slam_results(tmp_path: pathlib.Path):
    """
    Rescan is the only thing that pulls a finished run's results in, so it must leave no pane stale.

    Nothing polls any more. A pipeline writes its per-episode SLAM attrs to the store, and the
    operator presses Rescan; if that only swapped the task list, the scene pane and Episodes
    column they are looking at would still show pre-run numbers with nothing to say otherwise.
    """
    rec, engine = _seed(tmp_path)
    scene_dir = rec / 'scene_2026-07-26_10-00-00_abcd'
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 1
    ep = root.require_group('episode_0')
    ep.attrs['session_dir'] = 'session_1'
    slam = ep.require_group('annotations').require_group('slam')
    slam.attrs['n_frames_total'] = 100
    slam.attrs['n_frames_lost'] = 90
    slam.attrs['tracking_ratio'] = 0.1
    slam.attrs['n_relocalization_events'] = 0

    client = TestClient(create_app(engine, recordings_dir=rec))
    resp = client.post('/rescan', data={'selected_scene_id': 'scene-1'})

    assert resp.status_code == 200
    # the scene pane and its Episodes column both came back as out-of-band swaps...
    assert 'id="detail-body" hx-swap-oob="true"' in resp.text
    assert 'id="col-episodes-body" class="col-body" hx-swap-oob="true"' in resp.text
    # ...and only those two: the pane's own sub-fragments are rendered inline, since an
    # out-of-band swap nested inside another one would be swapped twice, to two places.
    assert resp.text.count('hx-swap-oob') == 2
    # ...the pane kept its task dropdown (jinja iterates an undefined all_tasks as empty)...
    assert '>fold_towel</option>' in resp.text
    # ...and the measurement the run wrote is now cached on the row it renders from
    with DBSession(engine) as db:
        row = db.exec(select(SessionRow).where(SessionRow.dir.like('%session_1'))).one()
        assert row.slam_attrs_json is not None and '"tracking_ratio": 0.1' in row.slam_attrs_json


def test_rescan_without_a_selected_scene_returns_only_the_task_list(tmp_path: pathlib.Path):
    """No scene open (or a deleted one) means no pane to refresh — and no OOB swap to send."""
    rec, engine = _seed(tmp_path)
    client = TestClient(create_app(engine, recordings_dir=rec))

    assert 'hx-swap-oob' not in client.post('/rescan').text
    assert 'hx-swap-oob' not in client.post('/rescan', data={'selected_scene_id': 'nope'}).text

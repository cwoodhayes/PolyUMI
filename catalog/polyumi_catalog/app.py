"""
FastAPI app for the catalog browser.

Serves the four-column Miller layout (Tasks → Scenes → Episodes → Datasets) plus a
detail panel over the synced catalog DB. Selecting a row issues an HTMX request that
swaps the column to its right and out-of-band-updates the detail panel (Phase 1,
read-only). Phase 2 adds task create/rename and scene→task assignment: these are
plain HTML form posts (no HTMX) that write the authoritative scene.json + DB row,
then redirect back to ``/`` for a full reload — simple and correct over
choreographing OOB swaps across every affected column. Phase 2.5 adds per-session
MCAP export + Foxglove launch; unlike Phase 2 these stay within the detail panel
(no other column is affected), so they're plain HTMX POSTs that re-swap
``#detail-body`` in place rather than reloading the whole page. Phase 3 adds the
dataset builder: building a dataset is a plain form POST + redirect (same pattern as
Phase 2, since it affects the Datasets column rather than wherever the form lives),
but *populating* that form is driven by an "Add to dataset" button in the scene
detail pane — a scene the user is looking at, not one selected from a picker in the
form itself. The set of scenes added so far (the "draft") has nowhere else to live
between requests, so it's kept as plain in-memory state on ``app.state`` — this is a
single-user, single-process, localhost tool (§2 non-goals), so process-global state
is fine and avoids adding cookies/sessions or client-side state for one working list.
It does not survive a server restart; that's an accepted trade-off, not an oversight.
Phase 4 adds a "run full pipeline" button on the scene detail pane: unlike every
prior mutation this can take minutes (SLAM in particular), so it can't run inline on
the request thread. It runs on a plain ``threading.Thread`` (started, not joined) with
per-scene status kept on ``app.state.pp_runs`` — same in-memory-on-app.state pattern as
the dataset draft, same trade-off (lost on restart). Progress is shown by the detail
pane polling itself via ``hx-trigger="every ...s"`` while a run is in flight, but the
authoritative progress log is still whatever's printed to the terminal running
``polyumi-catalog serve`` — the pipeline logs through the same ``logging`` root logger
that process already configures, so nothing extra was needed for that.
"""

from __future__ import annotations

import pathlib
import threading

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine
from sqlmodel import Session as DBSession

from polyumi_catalog import mcap_tools, pp_status, queries, thumbnails
from polyumi_catalog.db import default_datasets_dir
from polyumi_catalog.dataset_builder import DatasetBuildError, build_dataset
from polyumi_catalog.models import Scene
from polyumi_catalog.models import Session as SessionRow
from polyumi_catalog.mutations import MutationError, assign_scene_task, create_task, rename_task
from polyumi_catalog.sync import sync_datasets, sync_recordings

_PKG_DIR = pathlib.Path(__file__).resolve().parent
_TEMPLATES_DIR = _PKG_DIR / 'templates'
_STATIC_DIR = _PKG_DIR / 'static'


def create_app(engine: Engine, recordings_dir: pathlib.Path | None = None) -> FastAPI:
    """
    Build the catalog browser app bound to ``engine``.

    If ``recordings_dir`` is given, the ``/rescan`` endpoint re-runs the sync scan
    against it; otherwise rescanning is disabled.
    """
    app = FastAPI(title='PolyUMI Catalog')
    app.state.pending_dataset_scene_ids = []
    app.state.pp_runs = {}  # scene_id -> {'status': 'running'|'error', 'error': str|None}
    app.mount('/static', StaticFiles(directory=_STATIC_DIR), name='static')
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def render(request: Request, template: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, template, ctx)

    def render_dataset_builder(request: Request, db: DBSession, *, oob: bool) -> HTMLResponse:
        pending_scenes = queries.scenes_by_ids(db, app.state.pending_dataset_scene_ids)
        return render(request, '_dataset_builder.html', pending_scenes=pending_scenes, oob=oob)

    def _scene_detail_with_run_state(db: DBSession, scene_id: str) -> dict:
        """scene_detail plus this process's live pipeline-run status, if any."""
        detail = queries.scene_detail(db, scene_id)
        run_state = app.state.pp_runs.get(scene_id)
        detail['pp_running'] = run_state is not None and run_state['status'] == 'running'
        detail['pp_error'] = run_state['error'] if run_state else None
        return detail

    @app.get('/', response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        with DBSession(engine) as db:
            tasks = queries.list_tasks(db)
            pending_scenes = queries.scenes_by_ids(db, app.state.pending_dataset_scene_ids)
        return render(
            request,
            'index.html',
            tasks=tasks,
            pending_scenes=pending_scenes,
            can_rescan=recordings_dir is not None,
            can_build_dataset=recordings_dir is not None,
        )

    @app.get('/select/task/{task_key}', response_class=HTMLResponse)
    def select_task(request: Request, task_key: str) -> HTMLResponse:
        with DBSession(engine) as db:
            scenes = queries.list_scenes(db, task_key)
            datasets = queries.list_datasets(db, task_key)
            detail = queries.task_detail(db, task_key)
        return render(
            request,
            'select_task.html',
            scenes=scenes,
            datasets=datasets,
            detail=detail,
            selected_task=task_key,
        )

    @app.get('/select/scene/{scene_id}', response_class=HTMLResponse)
    def select_scene(request: Request, scene_id: str) -> HTMLResponse:
        with DBSession(engine) as db:
            sessions = queries.list_sessions(db, scene_id)
            detail = _scene_detail_with_run_state(db, scene_id)
            all_tasks = queries.list_task_options(db)
        return render(
            request, 'select_scene.html', sessions=sessions, detail=detail, all_tasks=all_tasks, selected_scene=scene_id
        )

    @app.get('/select/session/{session_id}', response_class=HTMLResponse)
    def select_session(request: Request, session_id: str) -> HTMLResponse:
        with DBSession(engine) as db:
            detail = queries.session_detail(db, session_id)
        return render(request, '_detail.html', detail=detail, oob=False)

    @app.get('/sessions/{session_id}/thumbnail.jpg')
    def get_session_thumbnail(session_id: str):
        with DBSession(engine) as db:
            session = db.get(SessionRow, session_id)
        if session is None:
            return PlainTextResponse('No such session.', status_code=404)
        jpeg = thumbnails.session_thumbnail_jpeg(pathlib.Path(session.dir))
        if jpeg is None:
            return PlainTextResponse('No thumbnail available.', status_code=404)
        # session directories are immutable once synced, so the browser can cache indefinitely
        headers = {'Cache-Control': 'public, max-age=31536000, immutable'}
        return Response(content=jpeg, media_type='image/jpeg', headers=headers)

    @app.get('/select/dataset/{dataset_id}', response_class=HTMLResponse)
    def select_dataset(request: Request, dataset_id: int) -> HTMLResponse:
        with DBSession(engine) as db:
            detail = queries.dataset_detail(db, dataset_id)
        return render(request, '_detail.html', detail=detail, oob=False)

    @app.post('/rescan', response_class=HTMLResponse)
    def rescan(request: Request) -> HTMLResponse:
        if recordings_dir is not None:
            sync_recordings(recordings_dir, engine)
            sync_datasets(default_datasets_dir(recordings_dir), engine)
        with DBSession(engine) as db:
            tasks = queries.list_tasks(db)
        return render(request, '_tasks.html', tasks=tasks, oob=False)

    @app.post('/tasks')
    def post_create_task(name: str = Form(...)):
        with DBSession(engine) as db:
            try:
                create_task(db, name)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
        return RedirectResponse('/', status_code=303)

    @app.post('/tasks/{task_id}/rename')
    def post_rename_task(task_id: int, new_name: str = Form(...)):
        with DBSession(engine) as db:
            try:
                rename_task(db, task_id, new_name)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
        return RedirectResponse('/', status_code=303)

    @app.post('/scenes/{scene_id}/task')
    def post_assign_scene_task(scene_id: str, task_id: str = Form('')):
        with DBSession(engine) as db:
            try:
                assign_scene_task(db, scene_id, int(task_id) if task_id else None)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
        return RedirectResponse('/', status_code=303)

    @app.post('/datasets')
    def post_build_dataset(name: str = Form(...), scene_ids: list[str] = Form([])):
        if recordings_dir is None:
            return PlainTextResponse('Dataset export requires a recordings directory.', status_code=400)

        with DBSession(engine) as db:
            try:
                build_dataset(
                    db,
                    name=name,
                    task_id=None,
                    scene_ids=scene_ids,
                    output_dir=default_datasets_dir(recordings_dir),
                )
            except DatasetBuildError as err:
                return PlainTextResponse(str(err), status_code=400)
        app.state.pending_dataset_scene_ids.clear()
        return RedirectResponse('/', status_code=303)

    @app.post('/dataset-draft/add/{scene_id}', response_class=HTMLResponse)
    def post_dataset_draft_add(request: Request, scene_id: str) -> HTMLResponse:
        with DBSession(engine) as db:
            if db.get(Scene, scene_id) is None:
                return PlainTextResponse('No such scene.', status_code=404)
            if scene_id not in app.state.pending_dataset_scene_ids:
                app.state.pending_dataset_scene_ids.append(scene_id)
            return render_dataset_builder(request, db, oob=True)

    @app.post('/dataset-draft/remove/{scene_id}', response_class=HTMLResponse)
    def post_dataset_draft_remove(request: Request, scene_id: str) -> HTMLResponse:
        if scene_id in app.state.pending_dataset_scene_ids:
            app.state.pending_dataset_scene_ids.remove(scene_id)
        with DBSession(engine) as db:
            return render_dataset_builder(request, db, oob=True)

    @app.post('/sessions/{session_id}/export-mcap', response_class=HTMLResponse)
    def post_export_mcap(request: Request, session_id: str) -> HTMLResponse:
        with DBSession(engine) as db:
            session = db.get(SessionRow, session_id)
            if session is None:
                return PlainTextResponse('No such session.', status_code=404)
            scene = db.get(Scene, session.scene_id)
            if scene is None:
                return PlainTextResponse('No such scene.', status_code=404)
            try:
                mcap_tools.export_session_to_mcap(pathlib.Path(scene.dir), pathlib.Path(session.dir).name)
            except mcap_tools.McapError as err:
                return PlainTextResponse(str(err), status_code=400)
            detail = queries.session_detail(db, session_id)
        return render(request, '_detail.html', detail=detail, oob=False)

    @app.post('/sessions/{session_id}/open-foxglove')
    def post_open_foxglove(session_id: str):
        with DBSession(engine) as db:
            session = db.get(SessionRow, session_id)
            if session is None:
                return PlainTextResponse('No such session.', status_code=404)
            scene = db.get(Scene, session.scene_id)
            if scene is None:
                return PlainTextResponse('No such scene.', status_code=404)
            mcap_path = mcap_tools.mcap_path_for_session(pathlib.Path(scene.dir), pathlib.Path(session.dir).name)
            if mcap_path is None:
                return PlainTextResponse('No MCAP exported yet for this session.', status_code=400)
            try:
                mcap_tools.open_in_foxglove(mcap_path)
            except mcap_tools.McapError as err:
                return PlainTextResponse(str(err), status_code=400)
        return Response(status_code=204)

    @app.post('/scenes/{scene_id}/run-pp', response_class=HTMLResponse)
    def post_run_pp(request: Request, scene_id: str) -> HTMLResponse:
        with DBSession(engine) as db:
            scene = db.get(Scene, scene_id)
            if scene is None:
                return PlainTextResponse('No such scene.', status_code=404)
            scene_dir = pathlib.Path(scene.dir)

        run_state = app.state.pp_runs.get(scene_id)
        if run_state is None or run_state['status'] != 'running':
            app.state.pp_runs[scene_id] = {'status': 'running', 'error': None}

            def _run() -> None:
                # Broad except is intentional: this runs unattended on a background
                # thread, so any failure anywhere in the pipeline (many exception
                # types across many steps, including external SLAM/ffmpeg subprocess
                # failures) must be captured here or the run status is stuck at
                # "running" forever with no way to observe what happened.
                try:
                    pp_status.run_full_pipeline(scene_dir)
                    app.state.pp_runs[scene_id] = {'status': 'done', 'error': None}
                except Exception as exc:
                    app.state.pp_runs[scene_id] = {'status': 'error', 'error': str(exc)}

            threading.Thread(target=_run, daemon=True).start()

        with DBSession(engine) as db:
            detail = _scene_detail_with_run_state(db, scene_id)
        return render(request, '_detail.html', detail=detail, oob=False)

    @app.get('/scenes/{scene_id}/pp-poll', response_class=HTMLResponse)
    def get_pp_poll(request: Request, scene_id: str) -> HTMLResponse:
        with DBSession(engine) as db:
            detail = _scene_detail_with_run_state(db, scene_id)
        return render(request, '_detail.html', detail=detail, oob=False)

    return app

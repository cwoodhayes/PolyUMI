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
Reads/writes of that list are still guarded by ``app.state.pending_dataset_lock`` — a
single-user tool can still see two nearly-simultaneous requests (e.g. two browser tabs),
and an unguarded check-then-append could otherwise add a duplicate entry.
Phase 4 adds a "run full pipeline" button on the scene detail pane: unlike every
prior mutation this can take minutes (SLAM in particular), so it can't run inline on
the request thread. It runs on a plain ``threading.Thread`` (started, not joined) with
per-scene status kept on ``app.state.pp_runs`` — same in-memory-on-app.state pattern as
the dataset draft, same trade-off (lost on restart). Progress is shown by the detail
pane polling itself via ``hx-trigger="every ...s"`` while a run is in flight, but the
authoritative progress log is still whatever's printed to the terminal running
``polyumi-catalog serve`` — the pipeline logs through the same ``logging`` root logger
that process already configures, so nothing extra was needed for that. Starting a run
is a check-then-set on that shared dict, so it's guarded by ``app.state.pp_runs_lock``
(a plain ``threading.Lock``) to keep two near-simultaneous POSTs (e.g. a fast double
click before the button disables) from both passing the "not already running" check
and starting two pipeline runs against the same scene.zarr concurrently. The pane
actually renders up to two buttons over that one route: "Run full pipeline" (the
original; ``force=false``, skips steps already marked complete — hidden once every
step is done, since it would then be a no-op) and "Re-run pipeline" (``force=true``,
re-runs every step from scratch regardless of completion — only shown once at least
one step is complete, i.e. there's actually something it would discard). Both post to
the same ``run-pp`` route with an ``hx-vals``-supplied ``force`` field; the route
itself is agnostic to which button fired it. A forced run clears the scene's recorded
step completion before starting (``pp_status.reset_pp_status``), so the pane drops to
0/N and re-ticks as the run proceeds instead of sitting at "complete" throughout.
Phase 5 adds marking an episode unusable (excluded from dataset exports): a plain HTMX POST
from the session detail pane, same in-place-swap style as MCAP export, except it also
out-of-band-updates the Episodes column (``_detail_with_episodes_oob``) so that column's
grey-out reflects the change immediately without requiring the scene to be reselected.
Phase 6 adds editable notes for both scenes and sessions: an HTMX form POST that re-swaps
``#detail-body`` in place, same style as every other detail-pane mutation. Session notes is
the first Session field the catalog itself writes back to metadata.json rather than only
ever reading (see the Session model + mutations module docstrings) — everything else there
still mirrors what the Pi recorded.
Phase 7 adds an editable task description, same in-place ``#detail-body`` swap as notes.
Unlike task rename (Phase 2), a description edit doesn't affect the Tasks column or any
other selection on the page, so it doesn't need the full-page reload rename uses — it's
scoped to the detail pane like every Phase 6 field, just with no on-disk manifest to write
(tasks have none; see the mutations module docstring).
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

from polyumi_catalog import mcap_tools, pp_status, provenance, queries, thumbnails
from polyumi_catalog.db import default_datasets_dir
from polyumi_catalog.dataset_builder import DatasetBuildError, build_dataset
from polyumi_catalog.models import Scene
from polyumi_catalog.models import Session as SessionRow
from polyumi_catalog.mutations import (
    MutationError,
    assign_scene_task,
    create_task,
    rename_task,
    set_scene_notes,
    set_session_notes,
    set_session_pose_source,
    set_session_unusable,
    set_task_description,
)
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
    app.state.pending_dataset_lock = threading.Lock()
    app.state.pp_runs = {}  # scene_id -> {'status': 'running'|'done'|'error', 'error': str|None}
    app.state.pp_runs_lock = threading.Lock()
    app.mount('/static', StaticFiles(directory=_STATIC_DIR), name='static')
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    # Commit shas are shown abbreviated in several places; the full value stays in the
    # element's title attribute, so the templates need both forms of the same string.
    templates.env.filters['short_sha'] = provenance.short_sha

    def render(request: Request, template: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, template, ctx)

    def render_dataset_builder(request: Request, db: DBSession, *, oob: bool) -> HTMLResponse:
        with app.state.pending_dataset_lock:
            pending_ids = list(app.state.pending_dataset_scene_ids)
        pending_scenes = queries.scenes_by_ids(db, pending_ids)
        all_tasks = queries.list_task_options(db)
        return render(request, '_dataset_builder.html', pending_scenes=pending_scenes, all_tasks=all_tasks, oob=oob)

    def _detail_with_episodes_oob(db: DBSession, session_id: str) -> HTMLResponse:
        """
        Re-render the session detail pane plus an out-of-band update of its scene's Episodes column.

        This keeps a usable/unusable toggle's grey-out reflected immediately if that scene's
        episode list happens to be open (same "update a currently-visible widget" pattern as
        the dataset-draft builder — see the module docstring's Phase 3 paragraph).
        """
        detail = queries.session_detail(db, session_id)
        detail_html = templates.env.get_template('_detail.html').render(detail=detail, oob=False)
        if detail.get('scene_id'):
            sessions = queries.list_sessions(db, detail['scene_id'])
            episodes_html = templates.env.get_template('_episodes.html').render(sessions=sessions, oob=True)
        else:
            episodes_html = ''
        return HTMLResponse(detail_html + episodes_html)

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
            # unfiltered, so a dataset is visible right after POST /datasets redirects here —
            # the Datasets column is otherwise only filled by /select/task's OOB swap
            datasets = queries.list_datasets(db, queries.FILTER_ALL)
            with app.state.pending_dataset_lock:
                pending_ids = list(app.state.pending_dataset_scene_ids)
            pending_scenes = queries.scenes_by_ids(db, pending_ids)
            all_tasks = queries.list_task_options(db)
        return render(
            request,
            'index.html',
            tasks=tasks,
            datasets=datasets,
            pending_scenes=pending_scenes,
            all_tasks=all_tasks,
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

    @app.post('/tasks/{task_id}/description', response_class=HTMLResponse)
    def post_task_description(request: Request, task_id: int, description: str = Form('')) -> HTMLResponse:
        with DBSession(engine) as db:
            try:
                set_task_description(db, task_id, description)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
            detail = queries.task_detail(db, str(task_id))
        return render(request, '_detail.html', detail=detail, oob=False)

    @app.post('/scenes/{scene_id}/task')
    def post_assign_scene_task(scene_id: str, task_id: str = Form('')):
        with DBSession(engine) as db:
            try:
                assign_scene_task(db, scene_id, int(task_id) if task_id else None)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
        return RedirectResponse('/', status_code=303)

    @app.post('/datasets')
    def post_build_dataset(name: str = Form(...), task_id: str = Form(''), scene_ids: list[str] = Form([])):
        if recordings_dir is None:
            return PlainTextResponse('Dataset export requires a recordings directory.', status_code=400)

        with DBSession(engine) as db:
            try:
                build_dataset(
                    db,
                    name=name,
                    task_id=int(task_id) if task_id else None,
                    scene_ids=scene_ids,
                    output_dir=default_datasets_dir(recordings_dir),
                )
            except DatasetBuildError as err:
                return PlainTextResponse(str(err), status_code=400)
        with app.state.pending_dataset_lock:
            app.state.pending_dataset_scene_ids.clear()
        return RedirectResponse('/', status_code=303)

    @app.post('/dataset-draft/add/{scene_id}', response_class=HTMLResponse)
    def post_dataset_draft_add(request: Request, scene_id: str) -> HTMLResponse:
        with DBSession(engine) as db:
            if db.get(Scene, scene_id) is None:
                return PlainTextResponse('No such scene.', status_code=404)
            with app.state.pending_dataset_lock:
                if scene_id not in app.state.pending_dataset_scene_ids:
                    app.state.pending_dataset_scene_ids.append(scene_id)
            return render_dataset_builder(request, db, oob=True)

    @app.post('/dataset-draft/remove/{scene_id}', response_class=HTMLResponse)
    def post_dataset_draft_remove(request: Request, scene_id: str) -> HTMLResponse:
        with app.state.pending_dataset_lock:
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

    @app.post('/scenes/{scene_id}/notes', response_class=HTMLResponse)
    def post_scene_notes(request: Request, scene_id: str, notes: str = Form('')) -> HTMLResponse:
        with DBSession(engine) as db:
            try:
                set_scene_notes(db, scene_id, notes)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
            detail = _scene_detail_with_run_state(db, scene_id)
        return render(request, '_detail.html', detail=detail, oob=False)

    @app.post('/sessions/{session_id}/notes', response_class=HTMLResponse)
    def post_session_notes(request: Request, session_id: str, notes: str = Form('')) -> HTMLResponse:
        with DBSession(engine) as db:
            try:
                set_session_notes(db, session_id, notes)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
            detail = queries.session_detail(db, session_id)
        return render(request, '_detail.html', detail=detail, oob=False)

    def _post_mark_usable(session_id: str, unusable: bool) -> HTMLResponse:
        with DBSession(engine) as db:
            try:
                set_session_unusable(db, session_id, unusable)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
            return _detail_with_episodes_oob(db, session_id)

    @app.post('/sessions/{session_id}/mark-unusable', response_class=HTMLResponse)
    def post_mark_unusable(session_id: str) -> HTMLResponse:
        return _post_mark_usable(session_id, True)

    @app.post('/sessions/{session_id}/mark-usable', response_class=HTMLResponse)
    def post_mark_usable(session_id: str) -> HTMLResponse:
        return _post_mark_usable(session_id, False)

    @app.post('/sessions/{session_id}/pose-source', response_class=HTMLResponse)
    def post_pose_source(session_id: str, source: str = Form(...)) -> HTMLResponse:
        with DBSession(engine) as db:
            try:
                set_session_pose_source(db, session_id, source)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
            return _detail_with_episodes_oob(db, session_id)

    @app.post('/scenes/{scene_id}/run-pp', response_class=HTMLResponse)
    def post_run_pp(request: Request, scene_id: str, force: bool = Form(False)) -> HTMLResponse:
        with DBSession(engine) as db:
            scene = db.get(Scene, scene_id)
            if scene is None:
                return PlainTextResponse('No such scene.', status_code=404)
            scene_dir = pathlib.Path(scene.dir)

        with app.state.pp_runs_lock:
            run_state = app.state.pp_runs.get(scene_id)
            already_running = run_state is not None and run_state['status'] == 'running'
            if not already_running:
                app.state.pp_runs[scene_id] = {'status': 'running', 'error': None}

        if not already_running:
            if force:
                # Done here rather than only inside run_full_pipeline so the response
                # rendered below already shows 0/N: the thread hasn't necessarily reached
                # the reset by the time this request returns, and the pane would otherwise
                # keep claiming "complete" until the first poll three seconds later.
                # Idempotent — run_full_pipeline repeats it for non-HTTP callers.
                try:
                    pp_status.reset_pp_status(scene_dir)
                except Exception as exc:
                    with app.state.pp_runs_lock:
                        app.state.pp_runs[scene_id] = {'status': 'error', 'error': str(exc)}
                    with DBSession(engine) as db:
                        detail = _scene_detail_with_run_state(db, scene_id)
                    return render(request, '_detail.html', detail=detail, oob=False)

            def _run() -> None:
                # Broad except is intentional: this runs unattended on a background
                # thread, so any failure anywhere in the pipeline (many exception
                # types across many steps, including external SLAM/ffmpeg subprocess
                # failures) must be captured here or the run status is stuck at
                # "running" forever with no way to observe what happened.
                try:
                    pp_status.run_full_pipeline(scene_dir, force=force)
                    with app.state.pp_runs_lock:
                        app.state.pp_runs[scene_id] = {'status': 'done', 'error': None}
                except Exception as exc:
                    with app.state.pp_runs_lock:
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

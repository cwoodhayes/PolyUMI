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
Phase 8 adds a topbar "Fetch from Pi" button beside Rescan: ``pingest fetch``'s scene copy
(``PiFetch``, minus the CLI's confirm prompt and its GoPro-SD-card follow-up, which needs a
mounted card) on a background thread with progress in ``app.state.fetch``, polled by the
topbar status span — the Phase 4 pattern, one run at a time rather than one per scene. The
run ends with the same sync ``/rescan`` does, and the poll that observes completion carries
an out-of-band Tasks refresh so the fetched scenes are browsable without a second click.
Phase 9 adds scene deletion from the scene detail pane — the one destructive operation here,
so it's a plain form POST + redirect (Phase 2 style, since three columns change) gated behind
a native ``confirm()`` and refused outright while that scene's pipeline is running. The
disk/DB guards live in ``mutations.delete_scene``.
"""

from __future__ import annotations

import pathlib
import shutil
import threading

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine
from sqlmodel import Session as DBSession
from sqlmodel import select

from polyumi_catalog import mcap_tools, pp_status, provenance, queries, thumbnails
from polyumi_catalog.db import default_datasets_dir
from polyumi_catalog.dataset_builder import DatasetBuildError, build_dataset
from polyumi_catalog.models import Dataset, DatasetMember, Scene
from polyumi_catalog.models import Session as SessionRow
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
from polyumi_catalog.sync import sync_datasets, sync_recordings, sync_scene_quality
from polyumi_ingest.pi_fetch import DEFAULT_HOST, PiFetch

_PKG_DIR = pathlib.Path(__file__).resolve().parent
_TEMPLATES_DIR = _PKG_DIR / 'templates'
_STATIC_DIR = _PKG_DIR / 'static'

# Pi-fetch progress shown in the topbar. 'total' stays None until the remote listing
# comes back, which is what the template renders as the "listing…" phase.
_IDLE_FETCH = {'status': 'idle', 'total': None, 'done': 0, 'current': None, 'error': None}


def create_app(engine: Engine, recordings_dir: pathlib.Path | None = None, pi_host: str = DEFAULT_HOST) -> FastAPI:
    """
    Build the catalog browser app bound to ``engine``.

    If ``recordings_dir`` is given, the ``/rescan`` and ``/fetch`` endpoints re-run the
    sync scan / pull new scenes off ``pi_host`` into it; otherwise both are disabled.
    """
    app = FastAPI(title='PolyUMI Catalog')
    app.state.pending_dataset_scene_ids = []
    app.state.pending_dataset_lock = threading.Lock()
    app.state.pp_runs = {}  # scene_id -> {'status': 'running'|'done'|'error', 'error': str|None}
    app.state.pp_runs_lock = threading.Lock()
    app.state.fetch = _IDLE_FETCH.copy()
    app.state.fetch_lock = threading.Lock()
    app.mount('/static', StaticFiles(directory=_STATIC_DIR), name='static')
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    # Commit shas are shown abbreviated in several places; the full value stays in the
    # element's title attribute, so the templates need both forms of the same string.
    templates.env.filters['short_sha'] = provenance.short_sha
    # A global rather than per-render context: _detail.html is rendered from a dozen call sites
    # (and from get_template().render() ones with no request), and all of them want the same
    # answer — POST /scenes/{id}/delete rejects when there's no recordings dir, so the button
    # must not be offered either.
    templates.env.globals['can_delete'] = recordings_dir is not None

    def render(request: Request, template: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, template, ctx)

    def render_dataset_builder(request: Request, db: DBSession, *, oob: bool) -> HTMLResponse:
        with app.state.pending_dataset_lock:
            pending_ids = list(app.state.pending_dataset_scene_ids)
        pending_scenes = queries.scenes_by_ids(db, pending_ids)
        all_tasks = queries.list_task_options(db)
        return render(request, '_dataset_builder.html', pending_scenes=pending_scenes, all_tasks=all_tasks, oob=oob)

    def _with_episodes_oob(db: DBSession, detail: dict, scene_id: str | None) -> HTMLResponse:
        """
        Render a detail pane plus an out-of-band update of ``scene_id``'s Episodes column.

        Keeps a currently-visible episode list honest whenever something behind it changed —
        a usable/unusable toggle's grey-out, a finished pipeline's quality badges (same
        "update a currently-visible widget" pattern as the dataset-draft builder; see the
        module docstring's Phase 3 paragraph).
        """
        detail_html = templates.env.get_template('_detail.html').render(detail=detail, oob=False)
        if not scene_id:
            return HTMLResponse(detail_html)
        sessions = queries.list_sessions(db, scene_id)
        episodes_html = templates.env.get_template('_episodes.html').render(sessions=sessions, oob=True)
        return HTMLResponse(detail_html + episodes_html)

    def _detail_with_episodes_oob(db: DBSession, session_id: str) -> HTMLResponse:
        """Session detail, with its scene's Episodes column refreshed out of band."""
        detail = queries.session_detail(db, session_id)
        return _with_episodes_oob(db, detail, detail.get('scene_id'))

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
            # a finished run's ✓/✗ belongs to the page that was watching it, not to every
            # page load until the server restarts, so a fresh render starts from idle
            fetch=app.state.fetch if app.state.fetch['status'] == 'running' else _IDLE_FETCH,
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
    def rescan(request: Request, selected_scene_id: str | None = Form(None)) -> HTMLResponse:
        """
        Re-read the recordings tree into the DB, and refresh whatever is on screen from it.

        Rescan is how anything that changed on disk outside a pipeline run reaches the UI — a
        scene fetched from the Pi, a store rebuilt from the CLI — so it should leave no stale
        pane behind. (The pp-poll only covers a run started from *this* process, and only while
        its pane is open.) ``selected_scene_id`` comes from the hidden input the scene detail
        pane renders, via hx-include on the button; when a scene is open, its pane and its
        Episodes column are swapped out of band alongside the task list. Absent means nothing,
        or a session, is selected, and the task list alone is enough.
        """
        if recordings_dir is not None:
            sync_recordings(recordings_dir, engine)
            sync_datasets(default_datasets_dir(recordings_dir), engine)
        with DBSession(engine) as db:
            tasks_html = templates.env.get_template('_tasks.html').render(tasks=queries.list_tasks(db), oob=False)
            if not selected_scene_id:
                return HTMLResponse(tasks_html)
            detail = _scene_detail_with_run_state(db, selected_scene_id)
            if detail.get('kind') != 'scene':  # deleted out from under the open pane
                return HTMLResponse(tasks_html)
            # all_tasks is not optional here: jinja iterates an undefined name as empty, so
            # omitting it silently swaps in a scene pane whose Assign-task dropdown has lost
            # every option.
            detail_html = templates.env.get_template('_detail.html').render(
                detail=detail, all_tasks=queries.list_task_options(db), oob=True
            )
            sessions = queries.list_sessions(db, selected_scene_id)
            episodes_html = templates.env.get_template('_episodes.html').render(sessions=sessions, oob=True)
        return HTMLResponse(tasks_html + detail_html + episodes_html)

    @app.post('/fetch', response_class=HTMLResponse)
    def post_fetch(request: Request) -> HTMLResponse:
        """
        Start a background ``pingest fetch`` of every not-yet-local scene on the Pi.

        Same background-thread + status-on-app.state + poll-the-fragment shape as the
        pipeline runner (see the Phase 4 paragraph above); a fetch is minutes of ssh/tar,
        so it can't run on the request thread either. Unlike the CLI's ``fetch`` this does
        not chase the GoPro SD card afterwards — that needs a mounted card, so it stays a
        ``pingest fetch-gopro`` on the terminal.
        """
        if recordings_dir is None:
            return PlainTextResponse('Fetching requires a recordings directory.', status_code=400)

        with app.state.fetch_lock:
            already_running = app.state.fetch['status'] == 'running'
            if not already_running:
                app.state.fetch = _IDLE_FETCH | {'status': 'running'}

        def _run() -> None:
            # Broad except for the same reason as the pipeline thread: unattended, so an
            # ssh/tar failure has to land in the status dict or the topbar spins forever.
            partial: pathlib.Path | None = None
            try:
                pi = PiFetch(pi_host)
                # Per session, not per scene: a scene grows while it's being recorded, so its
                # directory existing locally doesn't mean it's complete. Same helper the CLI
                # `pingest fetch` uses, so the button and the terminal agree — and one ssh for
                # the whole tree, so `total` below lands promptly rather than after a handshake
                # per scene with the progress bar stuck at 0.
                todo = [
                    (name, session)
                    for name, sessions in pi.missing_sessions(recordings_dir).items()
                    for session in sessions
                ]
                with app.state.fetch_lock:
                    app.state.fetch |= {'total': len(todo)}
                for i, (name, session) in enumerate(todo, 1):
                    with app.state.fetch_lock:
                        app.state.fetch |= {'current': f'{name}/{session}'}
                    partial = recordings_dir / name / session
                    pi.copy_sessions(name, [session], recordings_dir)
                    partial = None
                    with app.state.fetch_lock:
                        app.state.fetch |= {'done': i, 'current': None}
                sync_recordings(recordings_dir, engine)
                sync_datasets(default_datasets_dir(recordings_dir), engine)
                with app.state.fetch_lock:
                    app.state.fetch |= {'status': 'done'}
            except Exception as exc:
                with app.state.fetch_lock:
                    failed = app.state.fetch['current']  # None unless a copy_sessions raised
                    app.state.fetch |= {'status': 'error', 'error': f'{failed}: {exc}' if failed else str(exc)}
                if partial is not None:
                    # tar extracts in place, so a transfer that died halfway leaves a partial
                    # session dir — and the missing-session filter above would then treat it as
                    # already fetched. Drop it so the retry is just pressing the button again.
                    # Only that one session: the scene around it holds sessions already
                    # transferred, plus scene.zarr, scene.json, and the SLAM atlas.
                    shutil.rmtree(partial, ignore_errors=True)

        if not already_running:
            # kept on app.state purely so tests can join it instead of sleeping
            app.state.fetch_thread = threading.Thread(target=_run, daemon=True)
            app.state.fetch_thread.start()
        return render(request, '_fetch.html', fetch=app.state.fetch)

    @app.get('/fetch-poll', response_class=HTMLResponse)
    def get_fetch_poll(request: Request) -> HTMLResponse:
        fetch = app.state.fetch
        html = templates.env.get_template('_fetch.html').render(fetch=fetch)
        if fetch['status'] == 'done' and fetch['total']:
            # the run's closing sync already put the new scenes in the DB; refresh the Tasks
            # column out-of-band so they show up without the user also hitting Rescan
            with DBSession(engine) as db:
                tasks = queries.list_tasks(db)
            html += templates.env.get_template('_tasks.html').render(tasks=tasks, oob=True)
        return HTMLResponse(html)

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

    @app.post('/scenes/{scene_id}/delete')
    def post_delete_scene(scene_id: str):
        """
        Delete a scene from disk and from the catalog, then reload the page.

        Full reload (the Phase 2 pattern) rather than an in-pane swap: the scene vanishes from
        the Scenes column, its episodes from the Episodes column, and its task's count from the
        Tasks column, so there is nothing left to swap in place. Refuses while this scene's
        pipeline is running — that thread is writing into the very directory being removed.
        """
        if recordings_dir is None:
            return PlainTextResponse('Deleting requires a recordings directory.', status_code=400)
        run_state = app.state.pp_runs.get(scene_id)
        if run_state is not None and run_state['status'] == 'running':
            return PlainTextResponse('Pipeline is still running on this scene.', status_code=400)
        with DBSession(engine) as db:
            try:
                delete_scene(db, scene_id, recordings_dir)
            except MutationError as err:
                return PlainTextResponse(str(err), status_code=400)
        return RedirectResponse('/', status_code=303)

    @app.post('/datasets')
    def post_build_dataset(
        name: str = Form(...),
        task_id: str = Form(''),
        scene_ids: list[str] = Form([]),
        exporter_type: str = Form('dp'),
    ):
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
                    exporter_type=exporter_type,
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

    @app.post('/dataset-draft/from-dataset/{dataset_id}', response_class=HTMLResponse)
    def post_dataset_draft_from_dataset(request: Request, dataset_id: int) -> HTMLResponse:
        """Replace the draft with an existing dataset's member scenes, to re-export it after export logic changes."""
        with DBSession(engine) as db:
            if db.get(Dataset, dataset_id) is None:
                return PlainTextResponse('No such dataset.', status_code=404)
            members = db.exec(select(DatasetMember).where(DatasetMember.dataset_id == dataset_id)).all()
            with app.state.pending_dataset_lock:
                app.state.pending_dataset_scene_ids = [m.scene_id for m in members]
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
                    # the run just wrote this scene's SLAM metrics; pull them into the DB the
                    # UI reads so its usable-episode counts don't sit stale until a rescan
                    sync_scene_quality(scene_id, engine)
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
    def get_pp_poll(scene_id: str) -> HTMLResponse:
        """
        Re-render only the parts of a running scene's pane that a run changes.

        Deliberately a cheap read — no disk, no re-sync. This fires every 3s for the whole run,
        so anything it touches gets touched hundreds of times. It reads the DB and renders; the
        fresh SLAM measurements it eventually shows are put there once, by the run's own
        ``sync_scene_quality`` when the pipeline finishes, and the next tick after that swaps
        them in. Resist re-syncing here to make the badges fill in progressively — that walks
        every episode's zarr attrs on every tick.

        Three fragments, not the whole pane: replacing ``#detail-body`` on a timer also
        replaced the Notes textarea, throwing away anything typed into it during a run.
        """
        with DBSession(engine) as db:
            detail = _scene_detail_with_run_state(db, scene_id)
            sessions = queries.list_sessions(db, scene_id)
        env = templates.env
        return HTMLResponse(
            env.get_template('_pp_panel.html').render(detail=detail)
            + env.get_template('_scene_quality.html').render(detail=detail, oob=True)
            + env.get_template('_episodes.html').render(sessions=sessions, oob=True)
        )

    return app

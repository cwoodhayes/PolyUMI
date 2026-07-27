"""
FastAPI app for the Phase 1 read-only catalog browser.

Serves the four-column Miller layout (Tasks → Scenes → Episodes → Datasets) plus a
detail panel over the synced catalog DB. Selecting a row issues an HTMX request that
swaps the column to its right and out-of-band-updates the detail panel; no route here
mutates disk or DB (mutations arrive in Phase 2).
"""

from __future__ import annotations

import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine
from sqlmodel import Session as DBSession

from polyumi_catalog import queries
from polyumi_catalog.sync import sync_recordings

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
    app.mount('/static', StaticFiles(directory=_STATIC_DIR), name='static')
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def render(request: Request, template: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, template, ctx)

    @app.get('/', response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        with DBSession(engine) as db:
            tasks = queries.list_tasks(db)
        return render(request, 'index.html', tasks=tasks, can_rescan=recordings_dir is not None)

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
            detail = queries.scene_detail(db, scene_id)
        return render(request, 'select_scene.html', sessions=sessions, detail=detail, selected_scene=scene_id)

    @app.get('/select/session/{session_id}', response_class=HTMLResponse)
    def select_session(request: Request, session_id: str) -> HTMLResponse:
        with DBSession(engine) as db:
            detail = queries.session_detail(db, session_id)
        return render(request, '_detail.html', detail=detail, oob=False)

    @app.get('/select/dataset/{dataset_id}', response_class=HTMLResponse)
    def select_dataset(request: Request, dataset_id: int) -> HTMLResponse:
        with DBSession(engine) as db:
            detail = queries.dataset_detail(db, dataset_id)
        return render(request, '_detail.html', detail=detail, oob=False)

    @app.post('/rescan', response_class=HTMLResponse)
    def rescan(request: Request) -> HTMLResponse:
        if recordings_dir is not None:
            sync_recordings(recordings_dir, engine)
        with DBSession(engine) as db:
            tasks = queries.list_tasks(db)
        return render(request, '_tasks.html', tasks=tasks, oob=False)

    return app

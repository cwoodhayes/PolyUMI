"""
Read-only query helpers over the catalog cache, for the Phase 1 browser.

These are pure functions of a DB session so they can be unit-tested without HTTP.
They return plain dicts (view models) rather than ORM rows so the templates stay
decoupled from the SQLModel schema.
"""

from __future__ import annotations

import pathlib

from sqlalchemy import func
from sqlmodel import Session as DBSession
from sqlmodel import select

from polyumi_catalog.models import Dataset, Scene, Session, Task

# Sentinel task filters used by the UI's pseudo-rows in the Tasks column.
FILTER_ALL = 'all'
FILTER_UNASSIGNED = 'unassigned'


def _basename(path: str | None) -> str:
    """Return the final path component of ``path`` (empty string if None)."""
    return pathlib.Path(path).name if path else ''


def _scene_counts_by_task(db: DBSession) -> dict[int | None, int]:
    """Map each ``task_id`` (None = unassigned) to its scene count."""
    rows = db.exec(select(Scene.task_id, func.count()).group_by(Scene.task_id)).all()
    return {task_id: n for task_id, n in rows}


def _episode_counts_by_scene(db: DBSession) -> dict[str, int]:
    """Map each scene_id to its EPISODE-session count."""
    rows = db.exec(
        select(Session.scene_id, func.count()).where(Session.session_type == 'EPISODE').group_by(Session.scene_id)
    ).all()
    return {scene_id: n for scene_id, n in rows}


def _session_counts_by_scene(db: DBSession) -> dict[str, int]:
    """Map each scene_id to its total session count."""
    rows = db.exec(select(Session.scene_id, func.count()).group_by(Session.scene_id)).all()
    return {scene_id: n for scene_id, n in rows}


def list_tasks(db: DBSession) -> list[dict]:
    """
    Return the Tasks column view models.

    The list is the ``all`` and ``unassigned`` pseudo-rows followed by every real
    task, each annotated with its scene count.
    """
    counts = _scene_counts_by_task(db)
    total = sum(counts.values())
    rows: list[dict] = [
        {'key': FILTER_ALL, 'name': 'All scenes', 'scene_count': total, 'pseudo': True},
        {'key': FILTER_UNASSIGNED, 'name': 'Unassigned', 'scene_count': counts.get(None, 0), 'pseudo': True},
    ]
    for task in db.exec(select(Task).order_by(Task.name)).all():
        rows.append(
            {
                'key': str(task.id),
                'id': task.id,
                'name': task.name,
                'description': task.description,
                'scene_count': counts.get(task.id, 0),
                'pseudo': False,
            }
        )
    return rows


def list_task_options(db: DBSession) -> list[dict]:
    """Return every real task as ``{'id', 'name'}``, for assignment dropdowns (no pseudo-rows)."""
    return [{'id': t.id, 'name': t.name} for t in db.exec(select(Task).order_by(Task.name)).all()]


def _scene_view(scene: Scene, task_name: str | None, ep_count: int, sess_count: int) -> dict:
    """Build a Scenes-column view model from a scene row and its counts."""
    return {
        'scene_id': scene.scene_id,
        'name': _basename(scene.dir) or scene.scene_id,
        'dir': scene.dir,
        'task_id': scene.task_id,
        'task_name': task_name,
        'archived': scene.archived,
        'episode_count': ep_count,
        'session_count': sess_count,
        'created_at': scene.created_at,
    }


def list_scenes(db: DBSession, task_key: str) -> list[dict]:
    """
    Return the Scenes column view models for a task filter.

    ``task_key`` is ``all``, ``unassigned``, or a stringified task id.
    """
    stmt = select(Scene)
    if task_key == FILTER_UNASSIGNED:
        stmt = stmt.where(Scene.task_id.is_(None))
    elif task_key != FILTER_ALL:
        stmt = stmt.where(Scene.task_id == int(task_key))
    scenes = db.exec(stmt.order_by(Scene.dir)).all()

    ep_counts = _episode_counts_by_scene(db)
    sess_counts = _session_counts_by_scene(db)
    task_names = {t.id: t.name for t in db.exec(select(Task)).all()}
    return [
        _scene_view(s, task_names.get(s.task_id), ep_counts.get(s.scene_id, 0), sess_counts.get(s.scene_id, 0))
        for s in scenes
    ]


def list_sessions(db: DBSession, scene_id: str) -> list[dict]:
    """Return the Episodes column view models (all sessions) for one scene."""
    sessions = db.exec(select(Session).where(Session.scene_id == scene_id).order_by(Session.dir)).all()
    return [
        {
            'session_id': s.session_id,
            'name': _basename(s.dir) or s.session_id,
            'session_type': s.session_type,
            'robot': s.robot,
            'duration_s': s.duration_s,
            'n_video_frames': s.n_video_frames,
            'video_dropped_frames': s.video_dropped_frames,
            'task_meta': s.task_meta,
            'created_at': s.created_at,
        }
        for s in sessions
    ]


def list_datasets(db: DBSession, task_key: str) -> list[dict]:
    """
    Return the Datasets column view models for a task filter.

    ``all``/``unassigned`` list every dataset; a task id filters to that task.
    """
    stmt = select(Dataset)
    if task_key not in (FILTER_ALL, FILTER_UNASSIGNED):
        stmt = stmt.where(Dataset.task_id == int(task_key))
    datasets = db.exec(stmt.order_by(Dataset.name)).all()
    return [
        {
            'id': d.id,
            'name': d.name,
            'task_id': d.task_id,
            'n_episodes': d.n_episodes,
            'output_path': d.output_path,
            'created_at': d.created_at,
        }
        for d in datasets
    ]


def task_detail(db: DBSession, task_key: str) -> dict:
    """Return the detail-panel view model for a Tasks-column selection."""
    if task_key == FILTER_ALL:
        total = sum(_scene_counts_by_task(db).values())
        return {'kind': 'task', 'name': 'All scenes', 'pseudo': True, 'scene_count': total}
    if task_key == FILTER_UNASSIGNED:
        return {
            'kind': 'task',
            'name': 'Unassigned',
            'pseudo': True,
            'scene_count': _scene_counts_by_task(db).get(None, 0),
        }
    task = db.get(Task, int(task_key))
    if task is None:
        return {'kind': 'empty'}
    return {
        'kind': 'task',
        'id': task.id,
        'name': task.name,
        'description': task.description,
        'pseudo': False,
        'scene_count': _scene_counts_by_task(db).get(task.id, 0),
        'created_at': task.created_at,
    }


def scene_detail(db: DBSession, scene_id: str) -> dict:
    """Return the detail-panel view model for a Scenes-column selection."""
    scene = db.get(Scene, scene_id)
    if scene is None:
        return {'kind': 'empty'}
    task = db.get(Task, scene.task_id) if scene.task_id is not None else None
    sessions = db.exec(select(Session).where(Session.scene_id == scene_id)).all()
    # scene.json vs metadata.json task disagreements (see sync.Conflict).
    conflicts = [
        {'session_id': s.session_id, 'task_meta': s.task_meta}
        for s in sessions
        if task is not None and s.task_meta and s.task_meta != task.name
    ]
    return {
        'kind': 'scene',
        'scene_id': scene.scene_id,
        'name': _basename(scene.dir) or scene.scene_id,
        'dir': scene.dir,
        'task_id': scene.task_id,
        'task_name': task.name if task else None,
        'notes': scene.notes,
        'archived': scene.archived,
        'created_at': scene.created_at,
        'synced_at': scene.synced_at,
        'n_sessions': len(sessions),
        'n_episodes': sum(1 for s in sessions if s.session_type == 'EPISODE'),
        'conflicts': conflicts,
    }


def session_detail(db: DBSession, session_id: str) -> dict:
    """Return the detail-panel view model for an Episodes-column selection."""
    s = db.get(Session, session_id)
    if s is None:
        return {'kind': 'empty'}
    return {
        'kind': 'session',
        'session_id': s.session_id,
        'name': _basename(s.dir) or s.session_id,
        'dir': s.dir,
        'scene_id': s.scene_id,
        'session_type': s.session_type,
        'robot': s.robot,
        'task_meta': s.task_meta,
        'duration_s': s.duration_s,
        'n_video_frames': s.n_video_frames,
        'video_dropped_frames': s.video_dropped_frames,
        'created_at': s.created_at,
    }


def dataset_detail(db: DBSession, dataset_id: int) -> dict:
    """Return the detail-panel view model for a Datasets-column selection."""
    d = db.get(Dataset, dataset_id)
    if d is None:
        return {'kind': 'empty'}
    task = db.get(Task, d.task_id) if d.task_id is not None else None
    return {
        'kind': 'dataset',
        'id': d.id,
        'name': d.name,
        'task_name': task.name if task else None,
        'n_episodes': d.n_episodes,
        'output_path': d.output_path,
        'manifest_path': d.manifest_path,
        'polyumi_version': d.polyumi_version,
        'created_at': d.created_at,
    }

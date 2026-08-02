"""
Mutating operations for the catalog UI: tasks, scene/session assignment, notes, and flags.

Every mutation writes the authoritative on-disk file first (``scene.json`` for
scene-level fields, a session's own ``metadata.json`` for session-level ones), then
updates the corresponding DB row in the same operation, per docs/catalog-ui-plan.md §3.1/§4
("task_id on scene is a cache of what scene.json says; the writer updates both
scene.json and the row in one operation") — session notes extend that same pattern to
metadata.json, which used to be a Pi-record-time-only, catalog-read-only file (see the
Session model docstring). Task rows themselves have no on-disk manifest of their own —
the canonical task list is recoverable by re-syncing, since every scene.json.task string
round-trips through _get_or_create_task.
"""

from __future__ import annotations

import pathlib
from contextlib import AbstractContextManager

from polyumi_pi.files.metadata import SessionMetadata
from sqlmodel import Session as DBSession
from sqlmodel import select

from polyumi_catalog.manifests import SceneManifest, set_episode_unusable, update_scene_manifest
from polyumi_catalog.models import Scene, Session, Task


class MutationError(ValueError):
    """A mutation was rejected due to invalid input (e.g. a duplicate task name)."""


def _edit_scene_manifest(scene: Scene) -> AbstractContextManager[SceneManifest]:
    """
    Open ``scene``'s scene.json for a locked read-modify-write (see polyumi_ingest.manifests).

    Ingest owns that lock and that read-modify-write, since it writes this file too — flagging
    an episode unusable when preprocessing fails. Passing ``scene.scene_id`` keeps a
    not-yet-written manifest seeded from the DB row rather than re-read off disk.
    """
    return update_scene_manifest(pathlib.Path(scene.dir), default_scene_id=scene.scene_id)


def _clean_text(text: str | None) -> str | None:
    """Strip ``text``, collapsing a blank/whitespace-only value to ``None``."""
    return text.strip() or None if text is not None else None


def _write_scene_task(scene: Scene, task_name: str | None) -> None:
    """Rewrite ``scene.json`` for ``scene`` with a new task name, preserving its other fields."""
    with _edit_scene_manifest(scene) as manifest:
        manifest.task = task_name


def create_task(db: DBSession, name: str, description: str | None = None) -> Task:
    """Create a task named ``name``, or return the existing one if it already exists."""
    name = name.strip()
    if not name:
        raise MutationError('Task name cannot be empty.')
    existing = db.exec(select(Task).where(Task.name == name)).first()
    if existing is not None:
        return existing
    task = Task(name=name, description=description or None)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def rename_task(db: DBSession, task_id: int, new_name: str) -> Task:
    """
    Rename a task, cascading the new name into every member scene's ``scene.json``.

    Raises :class:`MutationError` if the new name collides with a different task or
    ``task_id`` doesn't exist.
    """
    new_name = new_name.strip()
    if not new_name:
        raise MutationError('Task name cannot be empty.')
    task = db.get(Task, task_id)
    if task is None:
        raise MutationError(f'No such task: {task_id}')
    collision = db.exec(select(Task).where(Task.name == new_name)).first()
    if collision is not None and collision.id != task_id:
        raise MutationError(f'Task {new_name!r} already exists.')

    task.name = new_name
    db.add(task)
    for scene in db.exec(select(Scene).where(Scene.task_id == task_id)).all():
        _write_scene_task(scene, new_name)
        db.add(scene)

    db.commit()
    db.refresh(task)
    return task


def set_task_description(db: DBSession, task_id: int, description: str | None) -> Task:
    """
    Set a task's description, writing only the DB row.

    Unlike scene/session fields, tasks have no on-disk manifest of their own (see the module
    docstring), so there's no file to rewrite here. A blank/whitespace value clears it.
    """
    task = db.get(Task, task_id)
    if task is None:
        raise MutationError(f'No such task: {task_id}')
    task.description = _clean_text(description)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def set_session_unusable(db: DBSession, session_id: str, unusable: bool) -> Session:
    """Mark a session's episode usable/unusable, writing scene.json + the DB row."""
    session = db.get(Session, session_id)
    if session is None:
        raise MutationError(f'No such session: {session_id}')
    scene = db.get(Scene, session.scene_id)
    if scene is None:
        raise MutationError(f'No such scene: {session.scene_id}')

    # Keyed by the session directory's basename rather than its session_id: session directories
    # are immutable once synced (see app.py's thumbnail-caching comment for the same invariant),
    # so the name is stable, and DP export (buffer.py) only has the directory name available on
    # the pzarr episode group, not the session_id.
    set_episode_unusable(
        pathlib.Path(scene.dir),
        pathlib.Path(session.dir).name,
        unusable,
        default_scene_id=scene.scene_id,
    )
    session.unusable = unusable
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


#: Valid values POST'd to /sessions/{id}/pose-source. 'default' clears the override (falls back
#: to the episode's eef.attrs['default_source'] at export time), matching pose_source_overrides
#: not having an entry at all for this session.
_POSE_SOURCES = ('default', 'optitrack', 'slam')


def _write_session_pose_source(scene: Scene, session_dir_name: str, source: str) -> None:
    """Rewrite ``scene.json`` for ``scene``, setting/clearing ``session_dir_name``'s pose source."""
    with _edit_scene_manifest(scene) as manifest:
        overrides = dict(manifest.pose_source_overrides)
        if source == 'default':
            overrides.pop(session_dir_name, None)
        else:
            overrides[session_dir_name] = source
        manifest.pose_source_overrides = overrides


def set_session_pose_source(db: DBSession, session_id: str, source: str) -> Session:
    """
    Set (or clear, via ``'default'``) a session's DP-export pose-source override.

    Writes scene.json's ``pose_source_overrides`` + the DB row, mirroring
    ``set_session_unusable``. Rejects an unknown ``source`` value outright; when the episode's
    available pose sources are known (pzarr built, step 5 has run), also rejects overriding to
    a source the episode never computed — same guard ``resolve_pose_source`` applies at export
    time, surfaced here instead of silently accepted and failing later.
    """
    if source not in _POSE_SOURCES:
        raise MutationError(f'Unknown pose source {source!r}; must be one of {_POSE_SOURCES}.')
    session = db.get(Session, session_id)
    if session is None:
        raise MutationError(f'No such session: {session_id}')
    scene = db.get(Scene, session.scene_id)
    if scene is None:
        raise MutationError(f'No such scene: {session.scene_id}')

    if source != 'default':
        from polyumi_catalog.pzarr_inspect import available_pose_sources

        session_dirname = pathlib.Path(session.dir).name
        available = available_pose_sources(pathlib.Path(scene.dir), session_dirname)
        if available is not None and source not in available:
            raise MutationError(
                f'{session_dirname} only has pose source(s) {available} — run `pingest pp 5 '
                f'--force` if it should have {source!r} too.'
            )

    _write_session_pose_source(scene, pathlib.Path(session.dir).name, source)
    session.pose_source_override = None if source == 'default' else source
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def set_scene_notes(db: DBSession, scene_id: str, notes: str | None) -> Scene:
    """Set a scene's notes, writing scene.json + the DB row. A blank/whitespace value clears them."""
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise MutationError(f'No such scene: {scene_id}')
    notes = _clean_text(notes)

    with _edit_scene_manifest(scene) as manifest:
        manifest.notes = notes

    scene.notes = notes
    db.add(scene)
    db.commit()
    db.refresh(scene)
    return scene


def set_session_notes(db: DBSession, session_id: str, notes: str | None) -> Session:
    """
    Set a session's notes, rewriting its metadata.json + the DB row.

    Unlike every other Session field, notes is now catalog-editable rather than a pure
    read-only mirror of what the Pi wrote at record time (see the Session model docstring).
    A blank/whitespace value clears them.
    """
    session = db.get(Session, session_id)
    if session is None:
        raise MutationError(f'No such session: {session_id}')
    notes = _clean_text(notes)

    meta_path = pathlib.Path(session.dir) / 'metadata.json'
    try:
        meta = SessionMetadata.from_file(meta_path)
    except FileNotFoundError:
        raise MutationError(f'metadata.json missing on disk for session: {session_id}') from None
    meta.notes = notes
    meta.to_file()

    session.notes = notes
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def assign_scene_task(db: DBSession, scene_id: str, task_id: int | None) -> Scene:
    """Assign (or clear, if ``task_id`` is None) a scene's task, writing scene.json + the DB row."""
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise MutationError(f'No such scene: {scene_id}')
    task_name = None
    if task_id is not None:
        task = db.get(Task, task_id)
        if task is None:
            raise MutationError(f'No such task: {task_id}')
        task_name = task.name

    _write_scene_task(scene, task_name)
    scene.task_id = task_id
    db.add(scene)
    db.commit()
    db.refresh(scene)
    return scene

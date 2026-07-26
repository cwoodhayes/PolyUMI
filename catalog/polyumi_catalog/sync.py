"""
Scan the recordings tree and upsert its scenes/sessions/tasks into the catalog DB.

The scan is idempotent and mtime-gated: a scene is re-parsed only when its directory,
one of its session directories, or a ``metadata.json`` has changed since the last sync
(or when ``force`` is set). SQLite is a pure cache here — every fact is re-derivable from
disk. See docs/catalog-ui-plan.md §6.

Known limitation: mtime gating uses the newest mtime among the scene dir, its session
dirs, and their ``metadata.json`` files. In-place edits to other files a scene depends on
are not detected; use ``--force`` to rebuild unconditionally.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from polyumi_pi.files.metadata import SessionMetadata
from sqlmodel import Session as DBSession
from sqlmodel import select

from polyumi_catalog.manifests import SceneManifest
from polyumi_catalog.models import Scene, Session, Task

log = logging.getLogger('catalog.sync')


@dataclass
class Conflict:
    """A session whose own ``metadata.json`` task disagrees with the scene's canonical task."""

    session_id: str
    scene_id: str
    scene_task: str | None
    meta_task: str | None


@dataclass
class SyncStats:
    """Summary of a single sync run."""

    scenes_scanned: int = 0
    scenes_updated: int = 0
    scenes_skipped: int = 0
    sessions_upserted: int = 0
    sessions_removed: int = 0
    tasks_created: int = 0
    conflicts: list[Conflict] = field(default_factory=list)


def _utc_ts(dt: datetime) -> float:
    """POSIX timestamp for ``dt``, treating a naive datetime as UTC (SQLite returns naive)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _newest_mtime(scene_dir: pathlib.Path, session_dirs: list[pathlib.Path]) -> float:
    """Newest mtime among the scene dir, its session dirs, and their metadata.json files."""
    mtimes = [scene_dir.stat().st_mtime]
    for sd in session_dirs:
        mtimes.append(sd.stat().st_mtime)
        md = sd / 'metadata.json'
        if md.is_file():
            mtimes.append(md.stat().st_mtime)
    return max(mtimes)


def _is_archived(scene_dir: pathlib.Path) -> bool:
    """Report whether a scene is archived (a ``*.zarr.zip`` exists but no working ``scene.zarr``)."""
    has_zip = any(scene_dir.glob('*.zarr.zip'))
    has_working = (scene_dir / 'scene.zarr').is_dir()
    return has_zip and not has_working


def _get_or_create_task(db: DBSession, name: str, stats: SyncStats) -> Task:
    """Return the task row named ``name``, creating it (and counting it) if absent."""
    task = db.exec(select(Task).where(Task.name == name)).first()
    if task is None:
        task = Task(name=name)
        db.add(task)
        db.flush()  # assign task.id
        stats.tasks_created += 1
    return task


def _sync_scene(db: DBSession, scene_dir: pathlib.Path, now: datetime, force: bool, stats: SyncStats) -> None:
    """Upsert a single scene directory and its sessions into the DB."""
    session_dirs = sorted(d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith('session_'))
    manifest = SceneManifest.from_scene_dir(scene_dir)

    existing_scene = db.exec(select(Scene).where(Scene.dir == str(scene_dir))).first()
    if (
        not force
        and existing_scene is not None
        and existing_scene.synced_at is not None
        and _newest_mtime(scene_dir, session_dirs) <= _utc_ts(existing_scene.synced_at)
    ):
        stats.scenes_skipped += 1
        return

    # parse session metadata
    metas: list[tuple[pathlib.Path, SessionMetadata]] = []
    for sd in session_dirs:
        md_path = sd / 'metadata.json'
        if not md_path.is_file():
            continue
        try:
            metas.append((sd, SessionMetadata.from_file(md_path)))
        except Exception as err:
            log.error(f'Failed to parse {md_path}: {err}')

    # resolve scene identity + canonical task
    scene_id = manifest.scene_id if manifest else (metas[0][1].scene_id if metas else scene_dir.name)
    task_name = manifest.task if manifest else None
    task_id = _get_or_create_task(db, task_name, stats).id if task_name else None
    created_ats = [m.created_at for _, m in metas if m.created_at is not None]
    scene_created = min(created_ats) if created_ats else None

    scene = db.get(Scene, scene_id) or Scene(scene_id=scene_id)
    scene.dir = str(scene_dir)
    scene.task_id = task_id
    scene.notes = manifest.notes if manifest else scene.notes
    scene.archived = _is_archived(scene_dir)
    scene.created_at = scene_created
    scene.synced_at = now
    db.add(scene)

    # upsert sessions and record task conflicts
    seen: set[str] = set()
    for sd, meta in metas:
        seen.add(meta.session_id)
        row = db.get(Session, meta.session_id) or Session(session_id=meta.session_id)
        row.scene_id = scene_id
        row.dir = str(sd)
        row.session_type = meta.session_type.value
        row.task_meta = meta.task
        row.robot = meta.robot
        row.duration_s = meta.duration_s
        row.n_video_frames = meta.n_video_frames
        row.video_dropped_frames = meta.video_dropped_frames
        row.created_at = meta.created_at
        db.add(row)
        stats.sessions_upserted += 1

        if task_name and meta.task and meta.task != task_name:
            stats.conflicts.append(Conflict(meta.session_id, scene_id, task_name, meta.task))

    # reconcile: drop session rows that no longer exist on disk
    for row in db.exec(select(Session).where(Session.scene_id == scene_id)).all():
        if row.session_id not in seen:
            db.delete(row)
            stats.sessions_removed += 1

    stats.scenes_updated += 1


def sync_recordings(recordings_dir: pathlib.Path, engine, *, force: bool = False) -> SyncStats:
    """
    Scan ``recordings_dir`` for ``scene_*`` directories and upsert them into the catalog.

    Returns a :class:`SyncStats` describing what changed.
    """
    stats = SyncStats()
    now = datetime.now(timezone.utc)
    with DBSession(engine) as db:
        for scene_dir in sorted(recordings_dir.iterdir() if recordings_dir.is_dir() else []):
            if not (scene_dir.is_dir() and scene_dir.name.startswith('scene_')):
                continue
            stats.scenes_scanned += 1
            _sync_scene(db, scene_dir, now, force, stats)
        db.commit()
    return stats

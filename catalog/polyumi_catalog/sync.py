"""
Scan the recordings tree and upsert its scenes/sessions/tasks/datasets into the catalog DB.

The scene/session scan is idempotent and mtime-gated: a scene is re-parsed only when its
directory, one of its session directories, or a ``metadata.json`` has changed since the last
sync (or when ``force`` is set). SQLite is a pure cache here — every fact is re-derivable from
disk.

Known limitation: mtime gating uses the newest mtime among the scene dir, its session
dirs, and their ``metadata.json`` files. In-place edits to other files a scene depends on
are not detected; use ``--force`` to rebuild unconditionally.

The one exception is each episode's cached SLAM quality (``Session.slam_attrs_json``),
which is refreshed on **every** sync, gate or no gate. Preprocessing writes those attrs
deep inside ``scene.zarr``, where nothing the gate looks at changes, so a gated sync would
otherwise leave a freshly-SLAMmed scene showing its pre-run numbers forever. Re-reading
them costs one pzarr open per scene (~50 ms), which is affordable for an explicit sync and
is exactly what it buys: page renders then never touch pzarr at all.

Dataset sync (``sync_datasets``) is a separate, simpler pass: dataset manifests are written
once at export time and never edited in place, so it just re-parses every ``*.dataset.json``
under the datasets directory on every call — no mtime gating needed.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from polyumi_pi.files.metadata import SessionMetadata
from sqlmodel import Session as DBSession
from sqlmodel import select

from polyumi_catalog import episode_quality
from polyumi_catalog.manifests import DatasetManifest, SceneManifest
from polyumi_catalog.models import Dataset, DatasetMember, Scene, Session, Task

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
    sessions_requalified: int = 0
    tasks_created: int = 0
    conflicts: list[Conflict] = field(default_factory=list)


@dataclass
class DatasetSyncStats:
    """Summary of a single dataset-sync run."""

    datasets_scanned: int = 0
    datasets_updated: int = 0
    manifests_failed: int = 0
    tasks_created: int = 0


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


def _get_or_create_task(db: DBSession, name: str) -> tuple[Task, bool]:
    """Return the task row named ``name`` and whether it was just created."""
    task = db.exec(select(Task).where(Task.name == name)).first()
    if task is None:
        task = Task(name=name)
        db.add(task)
        db.flush()  # assign task.id
        return task, True
    return task, False


def _refresh_scene_quality(db: DBSession, scene_id: str, scene_dir: pathlib.Path, stats: SyncStats) -> None:
    """
    Mirror the scene's per-episode SLAM measurements from pzarr onto its session rows.

    Runs outside the mtime gate (see the module docstring). Rows whose cached values are
    already correct are left untouched, so a sync over an unchanged tree still writes
    nothing. A session with no measurements — no pzarr, SLAM not run yet, or an archived
    scene whose ``scene.zarr`` has been zipped away — is cleared back to ``None``, which
    reads as "unknown", not as a failure.
    """
    records = episode_quality.scene_slam_records(scene_dir)
    for row in db.exec(select(Session).where(Session.scene_id == scene_id)).all():
        record = records.get(pathlib.Path(row.dir).name)
        attrs_json = episode_quality.record_to_json(record) if record else None
        has_optitrack = record.has_optitrack if record else None
        if row.slam_attrs_json == attrs_json and row.slam_has_optitrack == has_optitrack:
            continue
        row.slam_attrs_json = attrs_json
        row.slam_has_optitrack = has_optitrack
        db.add(row)
        stats.sessions_requalified += 1


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
        _refresh_scene_quality(db, existing_scene.scene_id, scene_dir, stats)
        return

    # parse session metadata
    metas: list[tuple[pathlib.Path, SessionMetadata]] = []
    parse_failed_dirs: set[str] = set()
    for sd in session_dirs:
        md_path = sd / 'metadata.json'
        if not md_path.is_file():
            continue
        try:
            metas.append((sd, SessionMetadata.from_file(md_path)))
        except Exception as err:
            log.error(f'Failed to parse {md_path}: {err}')
            parse_failed_dirs.add(str(sd))

    # resolve scene identity + canonical task
    scene_id = manifest.scene_id if manifest else (metas[0][1].scene_id if metas else scene_dir.name)
    task_name = None
    if manifest and manifest.task:
        task_name = manifest.task.strip() or None
    task_id = None
    if task_name:
        task, created = _get_or_create_task(db, task_name)
        task_id = task.id
        if created:
            stats.tasks_created += 1
    created_ats = [m.created_at for _, m in metas if m.created_at is not None]
    scene_created = min(created_ats) if created_ats else None

    # retire a stale row left behind if this scene's resolved identity has changed since
    # the last sync (e.g. metadata previously failed to parse everywhere, so scene_id fell
    # back to the directory name; now that it parses, the real scene_id resolves
    # differently) — otherwise two rows would end up pointing at the same directory.
    for stale in db.exec(select(Scene).where(Scene.dir == str(scene_dir), Scene.scene_id != scene_id)).all():
        db.delete(stale)

    scene = db.get(Scene, scene_id) or Scene(scene_id=scene_id)
    scene.dir = str(scene_dir)
    scene.task_id = task_id
    scene.notes = manifest.notes if manifest else scene.notes
    scene.archived = _is_archived(scene_dir)
    scene.created_at = scene_created
    scene.synced_at = now
    db.add(scene)

    # upsert sessions and record task conflicts
    unusable_dirs = set(manifest.unusable_episodes) if manifest else set()
    pose_source_overrides = manifest.pose_source_overrides if manifest else {}
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
        row.unusable = sd.name in unusable_dirs
        row.pose_source_override = pose_source_overrides.get(sd.name)
        row.notes = meta.notes
        row.created_at = meta.created_at
        db.add(row)
        stats.sessions_upserted += 1

        if task_name and meta.task and meta.task != task_name:
            stats.conflicts.append(Conflict(meta.session_id, scene_id, task_name, meta.task))

    # reconcile: drop session rows whose directory is genuinely gone. A directory whose
    # metadata.json merely failed to parse this round is left alone rather than evicted,
    # since the underlying data hasn't actually been removed (see parse_failed_dirs above).
    for row in db.exec(select(Session).where(Session.scene_id == scene_id)).all():
        if row.session_id in seen or row.dir in parse_failed_dirs:
            continue
        db.delete(row)
        stats.sessions_removed += 1

    _refresh_scene_quality(db, scene_id, scene_dir, stats)
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


def sync_scene_quality(scene_id: str, engine) -> int:
    """
    Re-read one scene's SLAM measurements into its session rows; returns how many changed.

    For callers that just wrote new ones — the "run pipeline" button — so the usable-episode
    counts reflect the run without waiting for the next full sync. A scene whose row has gone
    missing is a no-op.
    """
    stats = SyncStats()
    with DBSession(engine) as db:
        scene = db.get(Scene, scene_id)
        if scene is None:
            return 0
        _refresh_scene_quality(db, scene_id, pathlib.Path(scene.dir), stats)
        db.commit()
    return stats.sessions_requalified


def _episodes_to_db(episodes: str | list[int]) -> str:
    """Serialize a manifest member's ``episodes`` value for the ``DatasetMember.episodes`` column."""
    return episodes if isinstance(episodes, str) else json.dumps(episodes)


def _sync_dataset_manifest(
    db: DBSession, manifest: DatasetManifest, manifest_path: pathlib.Path, stats: DatasetSyncStats
) -> None:
    """Upsert one dataset manifest's Dataset row and fully replace its DatasetMember rows."""
    task_id = None
    task_name = manifest.task.strip() if manifest.task else None
    if task_name:
        task, created = _get_or_create_task(db, task_name)
        task_id = task.id
        if created:
            stats.tasks_created += 1

    dataset = db.exec(select(Dataset).where(Dataset.name == manifest.name)).first() or Dataset(name=manifest.name)
    dataset.task_id = task_id
    dataset.manifest_path = str(manifest_path)
    dataset.output_path = str(manifest_path.parent / manifest.output) if manifest.output else None
    dataset.n_episodes = manifest.n_episodes
    dataset.polyumi_version = manifest.polyumi_version
    db.add(dataset)
    db.flush()  # assign dataset.id if new

    for existing in db.exec(select(DatasetMember).where(DatasetMember.dataset_id == dataset.id)).all():
        db.delete(existing)
    for member in manifest.members:
        episodes = _episodes_to_db(member.episodes)
        db.add(DatasetMember(dataset_id=dataset.id, scene_id=member.scene_id, episodes=episodes))

    stats.datasets_updated += 1


def sync_datasets(datasets_dir: pathlib.Path, engine) -> DatasetSyncStats:
    """
    Scan ``datasets_dir`` for ``*.dataset.json`` manifests and upsert them into the catalog.

    Every manifest is re-parsed on every call (no mtime gating — see module docstring), so a
    dropped-and-rebuilt DB always recovers every dataset from its manifest, matching the
    source-of-truth-on-disk principle already applied to scene->task associations (§3.1).
    """
    stats = DatasetSyncStats()
    if not datasets_dir.is_dir():
        return stats
    with DBSession(engine) as db:
        for manifest_path in sorted(datasets_dir.glob('*.dataset.json')):
            stats.datasets_scanned += 1
            try:
                manifest = DatasetManifest.from_file(manifest_path)
            except Exception as err:
                log.error(f'Failed to parse {manifest_path}: {err}')
                stats.manifests_failed += 1
                continue
            _sync_dataset_manifest(db, manifest, manifest_path, stats)
        db.commit()
    return stats

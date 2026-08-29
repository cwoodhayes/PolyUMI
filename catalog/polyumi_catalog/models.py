"""
SQLModel tables for the catalog cache.

This database is a *rebuildable index* — the authoritative homes of each fact are the
on-disk ``metadata.json`` (sessions), ``scene.json`` (scene→task, notes), and the
per-dataset manifest. Any row here can be
reconstructed by re-running ``sync`` against the recordings tree.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class Task(SQLModel, table=True):
    """A named task that scenes are collected for (e.g. ``fold_towel``)."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Scene(SQLModel, table=True):
    """
    A scene directory grouping one or more sessions.

    ``task_id`` caches the assignment authoritatively stored in the scene's
    ``scene.json``; ``notes`` likewise mirrors the manifest.
    """

    scene_id: str = Field(primary_key=True)
    dir: str = Field(index=True)
    task_id: int | None = Field(default=None, foreign_key='task.id', index=True)
    notes: str | None = None
    archived: bool = False
    created_at: datetime | None = None
    #: Wall-clock span of the whole collection run: ``started_at`` is the Pi's scene start
    #: (metadata.json's ``scene_started_at``, falling back to the first session for scenes
    #: recorded before that field existed), ``ended_at`` the end of the last session. The
    #: gap between the two and the summed session durations is the dead time.
    started_at: datetime | None = None
    ended_at: datetime | None = None
    synced_at: datetime | None = None


class Session(SQLModel, table=True):
    """
    A single recording session (one demo episode or a mapping pass).

    Mirrors the session's ``metadata.json`` — including ``notes``, which the catalog UI can
    now edit in place (rewriting metadata.json), not just the Pi at record time — except
    ``unusable`` and ``pose_source_override``, which cache the scene's ``scene.json``
    (``unusable_episodes`` / ``pose_source_overrides``, both keyed by session directory name)
    instead, same cache-of-the-manifest pattern as ``Scene.task_id``/``Scene.notes``.
    ``task_meta`` preserves the session's own collection-time task string so the UI can flag it
    when it disagrees with the canonical scene-level task.

    ``slam_attrs_json`` / ``slam_has_optitrack`` mirror the episode's ``annotations/slam`` attrs
    out of the scene's pzarr — measurements only. The usable/unusable *verdict* over them is
    still computed on every read (``episode_quality.quality_view``), so editing
    ``quality_thresholds.yaml`` reclassifies everything with no re-sync; caching the numbers
    just keeps rendering a column of scenes from opening every pzarr store on disk.
    """

    session_id: str = Field(primary_key=True)
    scene_id: str = Field(foreign_key='scene.scene_id', index=True)
    dir: str
    session_type: str
    task_meta: str | None = None
    robot: str | None = None
    duration_s: float | None = None
    n_video_frames: int | None = None
    video_dropped_frames: int | None = None
    unusable: bool = False
    #: Cache of scene.json's pose_source_overrides for this session's directory name — same
    #: cache-of-the-manifest pattern as `unusable`. None means "no override, use the episode's
    #: default_source at export time"; otherwise 'optitrack' or 'slam'.
    pose_source_override: str | None = None
    notes: str | None = None
    created_at: datetime | None = None
    #: Verbatim JSON of the episode's `annotations/slam` attrs; None means SLAM hasn't run for
    #: it (or the scene has no pzarr yet). Stored whole rather than as one column per metric so
    #: a new metric needs no schema change — nothing here is queried by SQL, only re-parsed.
    slam_attrs_json: str | None = None
    #: Whether the episode has an OptiTrack pose source, which exempts it from the SLAM-derived
    #: checks. Nullable so it can be added to an existing DB in place; None reads as False.
    slam_has_optitrack: bool | None = None


class Dataset(SQLModel, table=True):
    """A named, training-ready combination of scenes exported to a UMI ReplayBuffer."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    task_id: int | None = Field(default=None, foreign_key='task.id')
    manifest_path: str | None = None
    output_path: str | None = None
    n_episodes: int | None = None
    polyumi_version: str | None = None
    #: Which ``polyumi_ingest.export.dp`` entry point produced this dataset: ``'dp'``
    #: (visuomotor only) or ``'polyumi'`` (adds the contact-mic and finger-camera
    #: modalities). Mirrors
    #: ``DatasetManifest.exporter_type``, which is the source of truth on re-sync.
    exporter_type: str = 'dp'
    #: Time accounting for the export, mirroring ``DatasetManifest``: wall-clock span of every
    #: member scene, recorded length of the sessions that made it in, and seconds actually in
    #: the buffer after trimming and segmentation.
    scene_seconds: float | None = None
    episode_seconds: float | None = None
    exported_seconds: float | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class DatasetMember(SQLModel, table=True):
    """
    One scene's membership in a dataset.

    ``episodes`` is ``"all"`` (whole-scene membership, the Phase 3 default) or a JSON
    list of episode indices once episode-level selection lands.
    """

    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key='dataset.id', index=True)
    scene_id: str = Field(foreign_key='scene.scene_id', index=True)
    episodes: str = 'all'

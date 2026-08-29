"""
How much wall-clock time a scene cost, and how much of it survived into a dataset.

Three numbers, narrowing: the *scene span* (the operator's whole run at the rig, dead time
between episodes included), the *episode* length of the sessions that made it into an export,
and the seconds actually in the buffer once the export's chirp trim and pose-gap segmentation
have had their say. Together they say how productive a collection run was. The catalog's scene
pane names its numbers the same way, and gets its span from :func:`span_from_metas` too.

Read from ``metadata.json`` rather than the catalog DB so ``pingest export`` and the catalog's
dataset builder report identical numbers -- the DB is only a cache of these files anyway.
"""

from __future__ import annotations

import logging
import pathlib
from datetime import datetime, timedelta

from polyumi_ingest.pzarr.scene_files import SceneFiles
from polyumi_pi.files.metadata import SessionMetadata

log = logging.getLogger(__name__)


def _session_metas(scene_dir: pathlib.Path) -> list[SessionMetadata]:
    """Parse every session's ``metadata.json`` under ``scene_dir``, skipping unreadable ones."""
    metas = []
    for session_dir in sorted(scene_dir.glob('session_*')):
        path = session_dir / 'metadata.json'
        if not path.is_file():
            continue
        try:
            metas.append(SessionMetadata.from_file(path))
        except Exception as err:
            log.warning(f'{path}: unreadable, not counted towards scene time ({err})')
    return metas


def _scene_root(scene_path: pathlib.Path) -> pathlib.Path:
    """Scene directory for a path given as either the scene root or its ``scene.zarr``."""
    return SceneFiles.resolve_zarr_path(scene_path).parent


def span_from_metas(metas: list[SessionMetadata]) -> tuple[datetime | None, datetime | None]:
    """
    When a scene began and ended, from its sessions' metadata.

    The one rule for that; the catalog's ``Scene.started_at``/``ended_at`` columns are filled
    from here too, so a scene's span is the same number whoever asks.

    The start is the Pi's ``scene_started_at``, stamped before the first session, so the span
    covers the setup and every pause between episodes. Scenes recorded before that field
    existed fall back to their first session and simply lose that lead-in. The end is derived:
    the Pi writes nothing when a scene stops, so the last session's end is as late as we know
    the operator was still working.
    """
    starts = [m.scene_started_at for m in metas if m.scene_started_at is not None]
    starts += [m.created_at for m in metas]
    ends = [m.created_at + timedelta(seconds=m.duration_s) for m in metas if m.duration_s is not None]
    return (min(starts) if starts else None, max(ends) if ends else None)


def _span_seconds(metas: list[SessionMetadata]) -> float | None:
    start, end = span_from_metas(metas)
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def scene_span_seconds(scene_path: pathlib.Path) -> float | None:
    """Wall clock from the start of a scene to the end of its last session, or None if unknown."""
    return _span_seconds(_session_metas(_scene_root(scene_path)))


def dataset_time_totals(scene_paths: list[pathlib.Path], provenance: list[dict]) -> dict[str, float | None]:
    """
    Sum the three time totals for one export: scene span, recorded episodes, exported seconds.

    ``provenance`` is what the ``export.dp`` entry points return. Its records are per *segment*,
    not per session -- a pose gap splits one session into several -- so the episode total
    de-duplicates on (scene, session); summing session durations straight off the records would
    count a split session twice.
    """
    recorded: dict[tuple[str, str], float] = {}
    spans: list[float | None] = []
    for scene_path in scene_paths:
        root = _scene_root(scene_path)
        metas = _session_metas(root)
        spans.append(_span_seconds(metas))
        by_dir = {m.path.parent.name: m for m in metas}
        for record in provenance:
            if record.get('scene') != root.name:
                continue
            meta = by_dir.get(record.get('session'))
            if meta is not None and meta.duration_s is not None:
                recorded[(root.name, meta.path.parent.name)] = meta.duration_s

    return {
        # None rather than a partial sum: one member scene with no measurable span makes the
        # whole "time at the rig" unknowable, and 0 would read as measured.
        'scene_seconds': None if None in spans else sum(spans),
        'episode_seconds': sum(recorded.values()),
        'exported_seconds': sum(float(r.get('duration_s') or 0.0) for r in provenance),
    }

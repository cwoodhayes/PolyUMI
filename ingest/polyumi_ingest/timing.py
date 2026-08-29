"""
How much wall-clock time a scene cost, and how much of it survived into a dataset.

Three numbers, narrowing: the *scene span* (the operator's whole run at the rig, dead time
between episodes included), the *recorded* length of the sessions that made it into an export,
and the seconds actually in the buffer once the export's chirp trim and pose-gap segmentation
have had their say. Together they say how productive a collection run was.

Read from ``metadata.json`` rather than the catalog DB so ``pingest export`` and the catalog's
dataset builder report identical numbers -- the DB is only a cache of these files anyway.
"""

from __future__ import annotations

import logging
import pathlib
from datetime import timedelta

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


def scene_root(scene_path: pathlib.Path) -> pathlib.Path:
    """Scene directory for a path given as either the scene root or its ``scene.zarr``."""
    return SceneFiles.resolve_zarr_path(scene_path).parent


def scene_span_seconds(scene_path: pathlib.Path) -> float | None:
    """
    Wall clock from the start of a scene to the end of its last session, or None if unknown.

    The start is the Pi's ``scene_started_at``, stamped before the first session, so the span
    covers the setup and every pause between episodes. Scenes recorded before that field
    existed fall back to their first session and simply lose that lead-in. The end is derived:
    the Pi writes nothing when a scene stops, so the last session's end is as late as we know
    the operator was still working.
    """
    metas = _session_metas(scene_root(scene_path))
    starts = [m.scene_started_at for m in metas if m.scene_started_at is not None]
    starts += [m.created_at for m in metas if m.created_at is not None]
    ends = [m.created_at + timedelta(seconds=m.duration_s) for m in metas if m.duration_s is not None]
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds()


def dataset_time_totals(scene_paths: list[pathlib.Path], provenance: list[dict]) -> dict[str, float]:
    """
    Sum the three time totals for one export: scene span, recorded episodes, exported seconds.

    ``provenance`` is what the ``export.dp`` entry points return. Its records are per *segment*,
    not per session -- a pose gap splits one session into several -- so the recorded total
    de-duplicates on (scene, session); summing session durations straight off the records would
    count a split session twice.
    """
    recorded: dict[tuple[str, str], float] = {}
    for scene_path in scene_paths:
        root = scene_root(scene_path)
        by_dir = {m.path.parent.name: m for m in _session_metas(root)}
        for record in provenance:
            if record.get('scene') != root.name:
                continue
            meta = by_dir.get(record.get('session'))
            if meta is not None and meta.duration_s is not None:
                recorded[(root.name, meta.path.parent.name)] = meta.duration_s

    spans = [s for p in scene_paths if (s := scene_span_seconds(p)) is not None]
    return {
        'scene_seconds': sum(spans),
        'episode_seconds': sum(recorded.values()),
        'exported_seconds': sum(float(r.get('duration_s') or 0.0) for r in provenance),
    }

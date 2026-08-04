"""
Per-episode SLAM tracking-quality stats surfaced in the catalog UI (Phase 4).

The numbers come from the ``annotations/slam`` group attrs the SLAM preprocessing step
(``OrbSlam3Step``, ingest step 2) already writes into each episode's pzarr group — no new
computation, just plumbing an existing per-episode summary through to the catalog so
tracking coverage is visible without opening Foxglove.

This module is split in two, and the split matters:

* :func:`scene_slam_records` is the only thing here that touches disk. It runs at **sync**
  time, and ``sync.py`` mirrors what it returns onto the ``Session`` row. Rendering a
  column of scenes therefore opens no pzarr stores at all — doing that per render cost
  ~1 s for a 22-scene recordings tree, on every click in the Scenes column, and grew
  linearly with the corpus.
* :func:`quality_view` turns one session's cached metrics into the verdict and badges the
  UI shows, via ``polyumi_ingest.quality`` (thresholds in
  ``ingest/config/quality_thresholds.yaml``). This runs on **every read**, so editing a
  threshold still reclassifies every episode at once with no re-sync: only the raw
  measurements are cached, never the verdict. DP export calls the same policy functions,
  so this view and what actually exports agree.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass

import zarr
from polyumi_ingest import quality as iquality

log = logging.getLogger('catalog.episode_quality')


@dataclass(frozen=True)
class SlamRecord:
    """One episode's raw SLAM measurements, as cached on its ``Session`` row."""

    #: The episode's ``annotations/slam`` attrs, verbatim.
    attrs: dict
    #: Whether the episode has an OptiTrack pose source (exempt from the SLAM checks).
    has_optitrack: bool


def scene_slam_records(scene_dir: pathlib.Path) -> dict[str, SlamRecord]:
    """
    Read every episode's raw SLAM measurements in one pass, keyed by source session dirname.

    Returns ``{}`` if pzarr doesn't exist yet. An episode is omitted from the result (rather
    than given an empty record) if it has no ``session_dir`` attr or SLAM (step 2) hasn't run
    for it yet — "nothing measured", which the verdict treats as "nothing to condemn it with".
    """
    zarr_path = scene_dir / 'scene.zarr'
    if not zarr_path.is_dir():
        return {}
    root = zarr.open_group(str(zarr_path), mode='r')
    n_episodes = int(root.attrs.get('n_episodes', 0))
    out: dict[str, SlamRecord] = {}
    for i in range(n_episodes):
        ep_key = f'episode_{i}'
        if ep_key not in root:
            continue
        ep = root[ep_key]
        session_dir = ep.attrs.get('session_dir')
        if not session_dir:
            continue
        if 'annotations' not in ep or 'slam' not in ep['annotations']:
            continue
        # OptiTrack episodes don't depend on SLAM for their pose source, so they're
        # exempt from the SLAM-derived checks. 'available_sources' is written by
        # step 5 (EefPoseStep); absent means step 5 hasn't run, i.e. not exempt.
        has_optitrack = 'optitrack' in list(ep['eef'].attrs.get('available_sources', [])) if 'eef' in ep else False
        out[session_dir] = SlamRecord(attrs=dict(ep['annotations']['slam'].attrs), has_optitrack=has_optitrack)
    return out


def record_to_json(record: SlamRecord) -> str:
    """Serialize a record's attrs for the ``Session.slam_attrs_json`` column."""
    return json.dumps(record.attrs)


def record_from_json(attrs_json: str | None, has_optitrack: bool | None) -> SlamRecord | None:
    """
    Rebuild a record from a ``Session`` row's cached columns, or ``None`` if it has none.

    Unparseable JSON is treated as "no measurements" rather than raising: the column is a
    cache, and one corrupt row shouldn't take down the page that lists its scene.
    """
    if not attrs_json:
        return None
    try:
        attrs = json.loads(attrs_json)
    except ValueError as err:
        log.warning(f'Ignoring unparseable cached SLAM attrs: {err}')
        return None
    return SlamRecord(attrs=attrs, has_optitrack=bool(has_optitrack))


def quality_view(record: SlamRecord | None) -> dict | None:
    """
    Apply the current thresholds to one episode's measurements, for display.

    Returns ``None`` when there are no measurements (SLAM hasn't run, no pzarr yet) — the
    callers render that as "unknown", not as a failing episode.
    """
    if record is None:
        return None
    thresholds = iquality.load_quality_thresholds()
    reasons = iquality.auto_unusable_reasons(record.attrs, has_optitrack=record.has_optitrack, thresholds=thresholds)
    return {
        'n_frames_total': record.attrs.get('n_frames_total'),
        'n_frames_lost': record.attrs.get('n_frames_lost'),
        'tracking_ratio': record.attrs.get('tracking_ratio'),
        'n_relocalization_events': record.attrs.get('n_relocalization_events'),
        'has_optitrack': record.has_optitrack,
        'low_quality': iquality.is_low_quality(record.attrs, thresholds=thresholds),
        #: Derived from the thresholds, not stored. An episode can also be unusable
        #: because a human listed it in scene.json; that's merged in by queries.py.
        'auto_unusable': bool(reasons),
        'auto_unusable_reasons': reasons,
    }


def scene_quality_by_session_dir(scene_dir: pathlib.Path) -> dict[str, dict]:
    """
    Read a scene's episode quality straight off disk, keyed by source session dirname.

    The catalog itself goes through the cached columns instead (see the module docstring);
    this is for callers with only a directory in hand, and for the export-side test that
    checks the catalog's verdict matches what DP export skips.
    """
    return {d: quality_view(r) for d, r in scene_slam_records(scene_dir).items()}


def scene_quality_summary(views: list[dict | None]) -> dict:
    """
    Aggregate SLAM tracking quality across a scene's episodes.

    Takes the per-episode :func:`quality_view` results; only episodes that actually have
    SLAM results (a non-``None`` view) contribute. If none do, returns an all-empty summary
    rather than raising.
    """
    rows = [v for v in views if v is not None]
    if not rows:
        return {
            'n_episodes_with_slam': 0,
            'avg_tracking_ratio': None,
            'n_low_quality': 0,
            'n_auto_unusable': 0,
        }
    ratios = [r['tracking_ratio'] for r in rows if r['tracking_ratio'] is not None]
    return {
        'n_episodes_with_slam': len(rows),
        'avg_tracking_ratio': sum(ratios) / len(ratios) if ratios else None,
        'n_low_quality': sum(1 for r in rows if r['low_quality']),
        'n_auto_unusable': sum(1 for r in rows if r['auto_unusable']),
    }

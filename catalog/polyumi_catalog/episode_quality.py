"""
Per-episode SLAM tracking-quality stats surfaced in the catalog UI (Phase 4).

Reads the ``annotations/slam`` group attrs the SLAM preprocessing step
(``OrbSlam3Step``, ingest step 2) already writes into each episode's pzarr
group — no new computation, just plumbing an existing per-episode summary
through to the catalog so tracking coverage is visible without opening
Foxglove. See docs/catalog-ui-plan.md Phase 4.
"""

from __future__ import annotations

import pathlib

import zarr

# below this fraction of tracked frames, an episode is flagged as low quality
LOW_TRACKING_RATIO = 0.9


def scene_quality_by_session_dir(scene_dir: pathlib.Path) -> dict[str, dict]:
    """
    Read every episode's SLAM quality in one pass, keyed by source session dirname.

    Returns ``{}`` if pzarr doesn't exist yet. An episode is omitted from the result
    (rather than given a ``None``-filled entry) if it has no ``session_dir`` attr or
    SLAM (step 2) hasn't run for it yet.
    """
    zarr_path = scene_dir / 'scene.zarr'
    if not zarr_path.is_dir():
        return {}
    root = zarr.open_group(str(zarr_path), mode='r')
    n_episodes = int(root.attrs.get('n_episodes', 0))
    out: dict[str, dict] = {}
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
        attrs = ep['annotations']['slam'].attrs
        tracking_ratio = attrs.get('tracking_ratio')
        out[session_dir] = {
            'n_frames_total': attrs.get('n_frames_total'),
            'n_frames_lost': attrs.get('n_frames_lost'),
            'tracking_ratio': tracking_ratio,
            'n_relocalization_events': attrs.get('n_relocalization_events'),
            'low_quality': tracking_ratio is not None and tracking_ratio < LOW_TRACKING_RATIO,
        }
    return out


def session_quality(scene_dir: pathlib.Path, session_dirname: str) -> dict | None:
    """Return one session's SLAM tracking-quality stats, or ``None`` if unavailable."""
    return scene_quality_by_session_dir(scene_dir).get(session_dirname)


def scene_quality_summary(scene_dir: pathlib.Path, session_dirnames: list[str]) -> dict:
    """
    Aggregate SLAM tracking quality across a scene's given session dirnames.

    Only sessions that actually have SLAM results contribute; if none do (SLAM
    hasn't run, or pzarr doesn't exist yet), returns an all-empty summary rather
    than raising.
    """
    by_dir = scene_quality_by_session_dir(scene_dir)
    rows = [by_dir[d] for d in session_dirnames if d in by_dir]
    if not rows:
        return {'n_episodes_with_slam': 0, 'avg_tracking_ratio': None, 'n_low_quality': 0}
    ratios = [r['tracking_ratio'] for r in rows if r['tracking_ratio'] is not None]
    return {
        'n_episodes_with_slam': len(rows),
        'avg_tracking_ratio': sum(ratios) / len(ratios) if ratios else None,
        'n_low_quality': sum(1 for r in rows if r['low_quality']),
    }

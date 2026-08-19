"""
Automatic episode-usability verdicts derived from stored SLAM quality metrics.

The metrics themselves live in each episode's ``annotations/slam`` attrs, written by
``OrbSlam3Step`` (preprocessing step 2). This module turns those numbers into a
usable/unusable verdict using thresholds from ``config/quality_thresholds.yaml``.

Two deliberate properties:

* **Derived, never stored.** No verdict is written into the pzarr, so editing the
  threshold file reclassifies every scene at once with no reprocessing. The pzarr
  holds measurements; this module holds policy.
* **One rule, two consumers.** The catalog UI and DP export both call
  ``auto_unusable_reasons``, so what the UI marks unusable is exactly what export
  skips.

An episode listed explicitly in ``scene.json``'s ``unusable_episodes`` is unusable
regardless of these thresholds — that set is a human decision and this module only
adds to it.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Mapping

import yaml

from polyumi_ingest.config import QUALITY_THRESHOLDS_YAML


@dataclass(frozen=True)
class QualityThresholds:
    """Thresholds controlling automatic unusable-flagging. See config/quality_thresholds.yaml."""

    max_lost_frames: int = 10
    min_tracked_frames: int = 60
    optitrack_always_usable: bool = True
    low_tracking_ratio: float = 0.90
    max_pose_jump_m: float = 0.08


@functools.lru_cache(maxsize=1)
def load_quality_thresholds() -> QualityThresholds:
    """
    Load thresholds from ``config/quality_thresholds.yaml``, falling back to defaults.

    Cached, since this is read on every catalog page render and every export. Call
    ``load_quality_thresholds.cache_clear()`` after editing the file in-process
    (tests do this).
    """
    try:
        with QUALITY_THRESHOLDS_YAML.open() as fh:
            raw = yaml.safe_load(fh) or {}
    except OSError:
        return QualityThresholds()
    fields = {f for f in QualityThresholds.__dataclass_fields__}
    return QualityThresholds(**{k: v for k, v in raw.items() if k in fields})


def _fed_frame_counts(slam_attrs: Mapping[str, Any]) -> tuple[int, int, str] | None:
    """
    Return ``(n_tracked, n_lost, window_label)`` on the fed grid, or None if unjudgeable.

    Prefers the post-chirp counts written since pzarr v4, which cover the span the exporter
    actually ships. Older stores only recorded whole-episode numbers, so fall back to those
    and label the window so the reason string can say which one it used — a v3 verdict is
    stricter than a v4 one on the same episode, since it counts the idle pre-chirp prefix
    where the localizer is still relocalizing.
    """
    n_fed = slam_attrs.get('n_frames_fed_post_chirp')
    n_lost = slam_attrs.get('n_frames_fed_lost_post_chirp')
    label = 'after the chirp'
    if not isinstance(n_fed, (int, float)) or not isinstance(n_lost, (int, float)):
        n_fed = slam_attrs.get('n_frames_fed')
        ratio = slam_attrs.get('tracking_ratio')
        if not isinstance(n_fed, (int, float)) or not isinstance(ratio, (int, float)) or math.isnan(ratio):
            return None
        # Pre-v4 stores recorded the ratio but not the tracked count; both are exact
        # integers underneath, so rounding the product recovers the count losslessly.
        n_lost = round(float(n_fed) * (1.0 - float(ratio)))
        label = 'whole episode, pre-v4 store'
    n_fed, n_lost = int(n_fed), int(n_lost)
    if n_fed <= 0:
        return None
    return n_fed - n_lost, n_lost, label


def auto_unusable_reasons(
    slam_attrs: Mapping[str, Any] | None,
    has_optitrack: bool = False,
    thresholds: QualityThresholds | None = None,
) -> list[str]:
    """
    Human-readable reasons this episode is automatically unusable; empty means usable.

    ``slam_attrs`` is an episode's ``annotations/slam`` attrs (``None``/empty when
    step 2 hasn't run — treated as "nothing to judge", i.e. usable, so an
    unprocessed scene isn't spuriously condemned).

    ``has_optitrack`` short-circuits every check when the thresholds allow it: such
    an episode's pose source doesn't depend on SLAM at all.

    The frame-count checks count frames SLAM was *fed*, not every GoPro frame — see
    ``_fed_frame_counts`` and the config file. Feeding the whole-grid ``n_frames_lost``
    to ``max_lost_frames`` would reject every episode processed at a stride above 1.

    ``max_pose_jump_m`` sits in the same mapping but is measured by step 5, on the hand-frame
    trajectory rather than reported by the localizer. It catches what the frame counts
    structurally cannot: an episode that tracked every frame it was fed and still teleported
    between two of them.
    """
    th = thresholds if thresholds is not None else load_quality_thresholds()
    if not slam_attrs:
        return []
    if has_optitrack and th.optitrack_always_usable:
        return []

    reasons: list[str] = []
    # Ahead of the frame counts, and deliberately not behind their `counts is None` bail: a
    # store too old to have post-chirp counts can still have a measured jump, and a metre-long
    # teleport is worth reporting on its own.
    jump = slam_attrs.get('max_pose_jump_m')
    if isinstance(jump, (int, float)) and not math.isnan(jump) and jump > th.max_pose_jump_m:
        reasons.append(
            f'{jump * 100:.0f} cm pose jump between adjacent frames (threshold {th.max_pose_jump_m * 100:.0f} cm)'
        )

    counts = _fed_frame_counts(slam_attrs)
    if counts is None:
        return reasons
    n_tracked, n_lost, window = counts

    if n_lost > th.max_lost_frames:
        reasons.append(f'{n_lost} frames lost {window} (threshold {th.max_lost_frames})')
    if n_tracked < th.min_tracked_frames:
        reasons.append(f'only {n_tracked} frames tracked {window} (threshold {th.min_tracked_frames})')
    return reasons


def is_low_quality(slam_attrs: Mapping[str, Any] | None, thresholds: QualityThresholds | None = None) -> bool:
    """Advisory 'low tracking' badge for the UI — separate from, and looser than, unusable."""
    th = thresholds if thresholds is not None else load_quality_thresholds()
    if not slam_attrs:
        return False
    ratio = slam_attrs.get('tracking_ratio')
    return isinstance(ratio, (int, float)) and not math.isnan(ratio) and ratio < th.low_tracking_ratio

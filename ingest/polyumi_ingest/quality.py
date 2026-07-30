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

    min_tracking_ratio: float = 0.80
    max_reverse_overlap_mm: float = 50.0
    optitrack_always_usable: bool = True
    low_tracking_ratio: float = 0.90


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

    A ``NaN`` self-consistency metric (fewer than 3 frames tracked by both passes,
    so the passes can't be compared) does *not* flag on its own — an episode that
    degenerate is caught by the coverage check instead.
    """
    th = thresholds if thresholds is not None else load_quality_thresholds()
    if not slam_attrs:
        return []
    if has_optitrack and th.optitrack_always_usable:
        return []

    reasons: list[str] = []

    ratio = slam_attrs.get('tracking_ratio')
    if isinstance(ratio, (int, float)) and not math.isnan(ratio) and ratio < th.min_tracking_ratio:
        reasons.append(f'only {ratio:.1%} of frames have a pose (threshold {th.min_tracking_ratio:.0%})')

    overlap = slam_attrs.get('reverse_overlap_median_mm')
    if isinstance(overlap, (int, float)) and not math.isnan(overlap) and overlap > th.max_reverse_overlap_mm:
        reasons.append(
            f'forward/reverse SLAM passes disagree by {overlap:.0f} mm (threshold {th.max_reverse_overlap_mm:.0f} mm)'
        )

    return reasons


def is_low_quality(slam_attrs: Mapping[str, Any] | None, thresholds: QualityThresholds | None = None) -> bool:
    """Advisory 'low tracking' badge for the UI — separate from, and looser than, unusable."""
    th = thresholds if thresholds is not None else load_quality_thresholds()
    if not slam_attrs:
        return False
    ratio = slam_attrs.get('tracking_ratio')
    return isinstance(ratio, (int, float)) and not math.isnan(ratio) and ratio < th.low_tracking_ratio

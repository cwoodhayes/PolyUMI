"""Tests for threshold-derived episode-usability verdicts (polyumi_ingest.quality)."""

from __future__ import annotations

import pathlib

from polyumi_ingest.quality import (
    QualityThresholds,
    auto_unusable_reasons,
    is_low_quality,
    load_quality_thresholds,
)

_TH = QualityThresholds(
    min_tracked_frames=60,
    optitrack_always_usable=True,
    low_tracking_ratio=0.90,
    max_pose_jump_m=0.08,
)


def _attrs(n_fed: int, n_lost: int) -> dict:
    """Post-chirp fed-grid counts as step 2 writes them since pzarr v4."""
    return {
        'n_frames_fed_post_chirp': n_fed,
        'n_frames_fed_lost_post_chirp': n_lost,
        'chirp_gated': True,
        # Whole-grid counts are also present on a real episode and must not be what's gated on:
        # under stride 2 half the grid is 'lost' by construction.
        'frame_stride': 2,
        'n_frames_total': n_fed * 2,
        'n_frames_lost': n_fed,
        'n_frames_fed': n_fed,
        'tracking_ratio': (n_fed - n_lost) / n_fed,
    }


def test_healthy_episode_is_usable() -> None:
    """Full coverage over the exported window: no reasons to exclude."""
    assert auto_unusable_reasons(_attrs(n_fed=220, n_lost=0), thresholds=_TH) == []


def test_an_episode_full_of_holes_is_still_usable() -> None:
    """
    Holes are segmentation's job, not a verdict's.

    This episode lost half its frames and is not excluded: the exporter splits a session at
    each dropout and keeps the runs either side, so whether it yields anything is decided by
    the segment floor, which only the exporter can evaluate.
    """
    assert auto_unusable_reasons(_attrs(n_fed=220, n_lost=110), thresholds=_TH) == []


def test_a_pose_jump_is_not_a_usability_verdict() -> None:
    """
    A teleport cuts the trajectory at export rather than condemning the demo.

    Real case from red_trapezoid_mug_v3: 205 frames fed, none lost, tracking_ratio 1.000, and
    a 1.14 m teleport between two adjacent frames — which is two episodes, not zero.
    """
    attrs = _attrs(n_fed=205, n_lost=0)
    attrs['max_pose_jump_m'] = 1.142
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []


def test_too_few_tracked_frames_flags_unusable() -> None:
    """The one surviving check: too short to contain a segment of the export's floor length."""
    reasons = auto_unusable_reasons(_attrs(n_fed=50, n_lost=0), thresholds=_TH)
    assert len(reasons) == 1
    assert 'only 50 frames tracked' in reasons[0]
    assert 'after the chirp' in reasons[0]


def test_tracked_frames_exactly_at_threshold_is_usable() -> None:
    """The bound is inclusive, so the boundary isn't off by one."""
    assert auto_unusable_reasons(_attrs(n_fed=60, n_lost=0), thresholds=_TH) == []


def test_whole_grid_losses_are_not_what_gets_counted() -> None:
    """
    A flawless stride-2 episode has half its whole-grid frames 'lost' by construction.

    Both counts must come off the *fed* grid. Subtracting the whole-grid `n_frames_lost` from
    the fed `n_frames_fed` mixes the two and reports zero tracked frames for a perfect
    episode — the trap asserted below, which would reject the entire corpus.
    """
    attrs = _attrs(n_fed=220, n_lost=0)
    assert attrs['n_frames_fed'] - attrs['n_frames_lost'] < _TH.min_tracked_frames  # the trap
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []


def test_optitrack_exempts_an_otherwise_unusable_episode() -> None:
    """An episode with OptiTrack poses doesn't depend on SLAM, so SLAM checks don't apply."""
    attrs = _attrs(n_fed=50, n_lost=0)
    assert auto_unusable_reasons(attrs, has_optitrack=True, thresholds=_TH) == []
    assert auto_unusable_reasons(attrs, has_optitrack=False, thresholds=_TH) != []


def test_optitrack_exemption_can_be_disabled_by_config() -> None:
    """With optitrack_always_usable off, OptiTrack episodes are judged like any other."""
    th = QualityThresholds(optitrack_always_usable=False, min_tracked_frames=60)
    assert auto_unusable_reasons(_attrs(n_fed=50, n_lost=0), has_optitrack=True, thresholds=th) != []


def test_missing_metrics_are_treated_as_nothing_to_judge() -> None:
    """No SLAM attrs at all (step 2 hasn't run) must not mark an episode unusable."""
    assert auto_unusable_reasons(None, thresholds=_TH) == []
    assert auto_unusable_reasons({}, thresholds=_TH) == []
    # Nor attrs too incomplete to derive a count from.
    assert auto_unusable_reasons({'n_frames_total': 400}, thresholds=_TH) == []


def test_pre_v4_store_falls_back_to_whole_episode_counts() -> None:
    """
    A store without the post-chirp attrs is judged on its whole-episode fed count.

    The reason string has to say so: the same episode is judged more harshly this way, since
    the idle pre-chirp prefix (where the localizer is still relocalizing) counts against it.
    """
    legacy = {'n_frames_fed': 50, 'tracking_ratio': 0.9}  # 45 tracked over the whole episode
    reasons = auto_unusable_reasons(legacy, thresholds=_TH)
    assert len(reasons) == 1
    assert 'only 45 frames tracked' in reasons[0]
    assert 'pre-v4' in reasons[0]


def test_pre_v4_fallback_recovers_exact_counts() -> None:
    """Deriving the tracked count from a float ratio must round-trip to the exact integer."""
    # 60/100 tracked = exactly at the inclusive bound.
    assert auto_unusable_reasons({'n_frames_fed': 100, 'tracking_ratio': 60 / 100}, thresholds=_TH) == []
    assert auto_unusable_reasons({'n_frames_fed': 100, 'tracking_ratio': 59 / 100}, thresholds=_TH) != []


def test_is_low_quality_is_looser_than_unusable() -> None:
    """The advisory badge fires on coverage without excluding the episode."""
    attrs = _attrs(n_fed=220, n_lost=8)
    attrs['tracking_ratio'] = 0.85
    assert is_low_quality(attrs, thresholds=_TH) is True
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []


def test_load_quality_thresholds_reads_the_shipped_config() -> None:
    """The shipped YAML parses and carries the documented defaults."""
    load_quality_thresholds.cache_clear()
    th = load_quality_thresholds()
    assert th.min_tracked_frames == 90
    assert th.max_pose_jump_m == 0.08
    assert th.optitrack_always_usable is True


def test_load_quality_thresholds_falls_back_when_config_missing(monkeypatch) -> None:
    """A missing config file yields defaults rather than raising and breaking the UI."""
    monkeypatch.setattr('polyumi_ingest.quality.QUALITY_THRESHOLDS_YAML', pathlib.Path('/nonexistent/x.yaml'))
    load_quality_thresholds.cache_clear()
    try:
        assert load_quality_thresholds() == QualityThresholds()
    finally:
        load_quality_thresholds.cache_clear()


def test_a_retired_key_in_a_local_config_is_inert(tmp_path, monkeypatch) -> None:
    """
    An unrecognised key is skipped, not passed into the dataclass ctor.

    This is what lets `max_lost_frames` be retired without breaking an operator's edited copy
    of the file: the key simply stops doing anything.
    """
    cfg = tmp_path / 'q.yaml'
    cfg.write_text('max_lost_frames: 5\nsomething_new: 12\nmin_tracked_frames: 42\n')
    monkeypatch.setattr('polyumi_ingest.quality.QUALITY_THRESHOLDS_YAML', cfg)
    load_quality_thresholds.cache_clear()
    try:
        th = load_quality_thresholds()
        assert th.min_tracked_frames == 42
        assert not hasattr(th, 'max_lost_frames')
    finally:
        load_quality_thresholds.cache_clear()

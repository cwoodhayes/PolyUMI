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
    max_lost_frames=10,
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


def test_whole_grid_losses_are_not_what_gets_gated() -> None:
    """
    The 219 whole-grid 'lost' frames of a flawless stride-2 episode must not condemn it.

    This is the failure mode the fed-grid counting exists to prevent: `n_frames_lost` counts
    every frame the localizer was never fed, so gating on it would reject the entire corpus.
    """
    attrs = _attrs(n_fed=220, n_lost=0)
    assert attrs['n_frames_lost'] > _TH.max_lost_frames  # the trap
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []


def test_too_many_lost_frames_flags_unusable() -> None:
    """Past the absolute lost-frame count the episode is dropped, UMI-style."""
    reasons = auto_unusable_reasons(_attrs(n_fed=220, n_lost=11), thresholds=_TH)
    assert len(reasons) == 1
    assert '11 frames lost' in reasons[0]
    assert 'after the chirp' in reasons[0]


def test_lost_frames_exactly_at_threshold_is_usable() -> None:
    """The bound is inclusive — exactly 10 lost passes, so the boundary isn't off by one."""
    assert auto_unusable_reasons(_attrs(n_fed=220, n_lost=10), thresholds=_TH) == []


def test_too_few_tracked_frames_flags_unusable() -> None:
    """A very short episode has few lost frames simply by being short; the floor catches it."""
    reasons = auto_unusable_reasons(_attrs(n_fed=50, n_lost=0), thresholds=_TH)
    assert len(reasons) == 1
    assert 'only 50 frames tracked' in reasons[0]


def test_both_failures_report_both_reasons() -> None:
    """The count and the floor are independent checks; both can fire at once."""
    assert len(auto_unusable_reasons(_attrs(n_fed=40, n_lost=20), thresholds=_TH)) == 2


def test_optitrack_exempts_an_otherwise_unusable_episode() -> None:
    """An episode with OptiTrack poses doesn't depend on SLAM, so SLAM checks don't apply."""
    attrs = _attrs(n_fed=100, n_lost=90)
    assert auto_unusable_reasons(attrs, has_optitrack=True, thresholds=_TH) == []
    assert auto_unusable_reasons(attrs, has_optitrack=False, thresholds=_TH) != []


def test_optitrack_exemption_can_be_disabled_by_config() -> None:
    """With optitrack_always_usable off, OptiTrack episodes are judged like any other."""
    th = QualityThresholds(optitrack_always_usable=False, max_lost_frames=10)
    assert auto_unusable_reasons(_attrs(n_fed=100, n_lost=90), has_optitrack=True, thresholds=th) != []


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
    legacy = {'n_frames_fed': 220, 'tracking_ratio': 0.9}  # 22 lost over the whole episode
    reasons = auto_unusable_reasons(legacy, thresholds=_TH)
    assert len(reasons) == 1
    assert '22 frames lost' in reasons[0]
    assert 'pre-v4' in reasons[0]


def test_pre_v4_fallback_recovers_exact_counts() -> None:
    """Deriving the lost count from a float ratio must round-trip to the exact integer."""
    assert auto_unusable_reasons({'n_frames_fed': 220, 'tracking_ratio': 1.0}, thresholds=_TH) == []
    # 210/220 tracked = 10 lost, exactly at the inclusive bound.
    assert auto_unusable_reasons({'n_frames_fed': 220, 'tracking_ratio': 210 / 220}, thresholds=_TH) == []
    assert auto_unusable_reasons({'n_frames_fed': 220, 'tracking_ratio': 209 / 220}, thresholds=_TH) != []


def test_a_pose_jump_condemns_an_otherwise_perfect_episode() -> None:
    """
    The blind spot the jump check exists for.

    Real case from red_trapezoid_mug_v3: an episode that was fed 205 frames, lost none of
    them, reported tracking_ratio 1.000 — and teleported 1.14 m between two adjacent frames.
    Every frame-count threshold calls this healthy, because no frame was lost.
    """
    attrs = _attrs(n_fed=205, n_lost=0)
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []

    attrs['max_pose_jump_m'] = 1.142
    reasons = auto_unusable_reasons(attrs, thresholds=_TH)
    assert len(reasons) == 1
    assert '114 cm pose jump' in reasons[0]


def test_normal_hand_motion_is_not_a_jump() -> None:
    """The clean corpus peaks at 6.2 cm; the threshold must sit above that, not in it."""
    attrs = _attrs(n_fed=205, n_lost=0)
    attrs['max_pose_jump_m'] = 0.062
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []
    # Inclusive bound, matching max_lost_frames.
    attrs['max_pose_jump_m'] = 0.08
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []


def test_pose_jump_is_judged_even_without_usable_frame_counts() -> None:
    """
    A store too old to derive fed-frame counts from still gets its jump judged.

    The check sits ahead of the counts precisely so it doesn't inherit their bail-out — a
    metre-long teleport is worth reporting on its own.
    """
    reasons = auto_unusable_reasons({'n_frames_total': 400, 'max_pose_jump_m': 3.06}, thresholds=_TH)
    assert len(reasons) == 1
    assert '306 cm pose jump' in reasons[0]


def test_optitrack_episodes_skip_the_jump_check_too() -> None:
    """The metric is measured on the SLAM trajectory, which such episodes don't export."""
    attrs = _attrs(n_fed=205, n_lost=0)
    attrs['max_pose_jump_m'] = 3.06
    assert auto_unusable_reasons(attrs, has_optitrack=True, thresholds=_TH) == []


def test_missing_jump_metric_condemns_nothing() -> None:
    """Episodes preprocessed before step 5 wrote the metric stay judged on frame counts alone."""
    assert auto_unusable_reasons(_attrs(n_fed=205, n_lost=0), thresholds=_TH) == []


def test_is_low_quality_is_looser_than_unusable() -> None:
    """The advisory badge fires on coverage without excluding the episode."""
    attrs = _attrs(n_fed=220, n_lost=8)  # 96.4% tracked... but set the ratio into the badge band
    attrs['tracking_ratio'] = 0.85
    assert is_low_quality(attrs, thresholds=_TH) is True
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []


def test_load_quality_thresholds_reads_the_shipped_config() -> None:
    """The shipped YAML parses and carries the documented defaults."""
    load_quality_thresholds.cache_clear()
    th = load_quality_thresholds()
    assert th.max_lost_frames == 10
    assert th.min_tracked_frames == 60
    assert th.optitrack_always_usable is True


def test_load_quality_thresholds_falls_back_when_config_missing(monkeypatch) -> None:
    """A missing config file yields defaults rather than raising and breaking the UI."""
    monkeypatch.setattr('polyumi_ingest.quality.QUALITY_THRESHOLDS_YAML', pathlib.Path('/nonexistent/x.yaml'))
    load_quality_thresholds.cache_clear()
    try:
        assert load_quality_thresholds() == QualityThresholds()
    finally:
        load_quality_thresholds.cache_clear()


def test_load_quality_thresholds_ignores_unknown_keys(tmp_path, monkeypatch) -> None:
    """An unrecognised key in the YAML is skipped, not passed into the dataclass ctor."""
    cfg = tmp_path / 'q.yaml'
    cfg.write_text('max_lost_frames: 5\nsomething_new: 12\n')
    monkeypatch.setattr('polyumi_ingest.quality.QUALITY_THRESHOLDS_YAML', cfg)
    load_quality_thresholds.cache_clear()
    try:
        assert load_quality_thresholds().max_lost_frames == 5
    finally:
        load_quality_thresholds.cache_clear()

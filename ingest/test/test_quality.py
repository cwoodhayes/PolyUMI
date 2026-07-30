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
    min_tracking_ratio=0.80,
    max_reverse_overlap_mm=50.0,
    optitrack_always_usable=True,
    low_tracking_ratio=0.90,
)


def test_healthy_episode_is_usable() -> None:
    """Good coverage and tight forward/reverse agreement: no reasons to exclude."""
    attrs = {'tracking_ratio': 1.0, 'reverse_overlap_median_mm': 0.55}
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []


def test_low_tracking_ratio_flags_unusable() -> None:
    """Below the coverage threshold, the episode is unusable and says so."""
    attrs = {'tracking_ratio': 0.645, 'reverse_overlap_median_mm': 1.0}
    reasons = auto_unusable_reasons(attrs, thresholds=_TH)
    assert len(reasons) == 1
    assert '64.5%' in reasons[0]
    assert '80%' in reasons[0]


def test_tracking_ratio_exactly_at_threshold_is_usable() -> None:
    """The threshold is inclusive — exactly 80% passes, so the boundary isn't off by one."""
    assert auto_unusable_reasons({'tracking_ratio': 0.80}, thresholds=_TH) == []


def test_self_consistency_failure_flags_unusable() -> None:
    """
    A large forward/reverse disagreement flags the episode even at full coverage.

    Modelled on the real 74ee episode_3: 100% of frames got a pose, but the two passes
    disagreed by 453 mm and the poses were ~220 mm wrong. Coverage alone can't catch this.
    """
    attrs = {'tracking_ratio': 1.0, 'reverse_overlap_median_mm': 453.85}
    reasons = auto_unusable_reasons(attrs, thresholds=_TH)
    assert len(reasons) == 1
    assert '454 mm' in reasons[0] or '453 mm' in reasons[0]


def test_both_failures_report_both_reasons() -> None:
    """Coverage and self-consistency are independent checks; both can fire at once."""
    attrs = {'tracking_ratio': 0.2, 'reverse_overlap_median_mm': 300.0}
    assert len(auto_unusable_reasons(attrs, thresholds=_TH)) == 2


def test_optitrack_exempts_an_otherwise_unusable_episode() -> None:
    """An episode with OptiTrack poses doesn't depend on SLAM, so SLAM checks don't apply."""
    attrs = {'tracking_ratio': 0.0, 'reverse_overlap_median_mm': 999.0}
    assert auto_unusable_reasons(attrs, has_optitrack=True, thresholds=_TH) == []
    # ...but the same episode without OptiTrack is excluded.
    assert auto_unusable_reasons(attrs, has_optitrack=False, thresholds=_TH) != []


def test_optitrack_exemption_can_be_disabled_by_config() -> None:
    """With optitrack_always_usable off, OptiTrack episodes are judged like any other."""
    th = QualityThresholds(optitrack_always_usable=False, min_tracking_ratio=0.80)
    assert auto_unusable_reasons({'tracking_ratio': 0.1}, has_optitrack=True, thresholds=th) != []


def test_nan_self_consistency_does_not_flag_on_its_own() -> None:
    """
    A NaN agreement metric means the passes shared too few frames to compare, not failure.

    Such an episode is caught by the coverage check instead; flagging on NaN would also
    condemn every episode processed before the reverse pass existed.
    """
    attrs = {'tracking_ratio': 1.0, 'reverse_overlap_median_mm': float('nan')}
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []


def test_missing_metrics_are_treated_as_nothing_to_judge() -> None:
    """No SLAM attrs at all (step 2 hasn't run) must not mark an episode unusable."""
    assert auto_unusable_reasons(None, thresholds=_TH) == []
    assert auto_unusable_reasons({}, thresholds=_TH) == []


def test_pre_reverse_pass_episode_judged_on_coverage_only() -> None:
    """Episodes from before the reverse pass have no agreement metric; coverage still applies."""
    assert auto_unusable_reasons({'tracking_ratio': 0.95}, thresholds=_TH) == []
    assert auto_unusable_reasons({'tracking_ratio': 0.10}, thresholds=_TH) != []


def test_is_low_quality_is_looser_than_unusable() -> None:
    """The advisory badge fires between the two thresholds without excluding the episode."""
    attrs = {'tracking_ratio': 0.85}  # under low_tracking_ratio 0.9, over min 0.8
    assert is_low_quality(attrs, thresholds=_TH) is True
    assert auto_unusable_reasons(attrs, thresholds=_TH) == []


def test_load_quality_thresholds_reads_the_shipped_config() -> None:
    """The shipped YAML parses and carries the documented defaults."""
    load_quality_thresholds.cache_clear()
    th = load_quality_thresholds()
    assert th.min_tracking_ratio == 0.80
    assert th.max_reverse_overlap_mm == 50.0
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
    cfg.write_text('min_tracking_ratio: 0.5\nsomething_new: 12\n')
    monkeypatch.setattr('polyumi_ingest.quality.QUALITY_THRESHOLDS_YAML', cfg)
    load_quality_thresholds.cache_clear()
    try:
        assert load_quality_thresholds().min_tracking_ratio == 0.5
    finally:
        load_quality_thresholds.cache_clear()

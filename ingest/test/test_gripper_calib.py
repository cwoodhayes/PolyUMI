"""
Tests for the closed-gripper width statistic.

The reason this is not just ``np.nanmin`` (which is what UMI does) is that a plain minimum cannot
tell a genuine closed dwell from one bad PnP solve — both are a small float. These tests pin that
distinction, because getting it wrong produces a plausible-looking number that is millimetres off
and lands directly in a robot command.
"""

import numpy as np
import pytest
from polyumi_ingest.gripper_calib import (
    DEFAULT_PERCENTILE,
    MIN_PLATEAU_SAMPLES,
    PLATEAU_TOL_M,
    closed_width_stats,
    format_report,
)


def _open_close_series(closed_m=0.005, open_m=0.090, cycles=3, dwell=200, travel=50):
    """Build a realistic recording: several full cycles, dwelling at each extreme."""
    parts = []
    for _ in range(cycles):
        parts.append(np.full(dwell, closed_m))
        parts.append(np.linspace(closed_m, open_m, travel))
        parts.append(np.full(dwell, open_m))
        parts.append(np.linspace(open_m, closed_m, travel))
    return np.concatenate(parts)


def test_recovers_the_closed_plateau():
    """The basic case: closed width lands on the dwell value, not somewhere up the travel ramp."""
    stats = closed_width_stats(_open_close_series(closed_m=0.0052))

    assert stats.closed_width_m == pytest.approx(0.0052, abs=1e-4)
    assert stats.max_m == pytest.approx(0.090, abs=1e-4)
    assert stats.plateau_is_convincing
    assert not stats.min_is_outlier


def test_a_single_bad_detection_does_not_move_the_answer():
    """
    The failure mode this statistic exists for.

    One frame reconstructing 3 mm low is exactly what UMI's nanmin would take verbatim, and the
    resulting offset would be wrong by that much everywhere.
    """
    series = _open_close_series(closed_m=0.005)
    series[7] = 0.002  # a stray PnP solve

    stats = closed_width_stats(series)

    assert stats.min_m == pytest.approx(0.002)
    assert stats.closed_width_m == pytest.approx(0.005, abs=1e-4), 'the outlier must not reach closed width'
    assert stats.min_is_outlier, 'and it must be reported as detached from the plateau'


def test_flags_a_recording_where_the_gripper_was_never_held_shut():
    """
    A ramp through the closed position is not a measurement of it.

    Without the plateau tally this yields a confident-looking number off the bottom of the sweep,
    with nothing in the output to suggest the recording was unusable.
    """
    stats = closed_width_stats(np.linspace(0.005, 0.090, 2000))

    assert not stats.plateau_is_convincing
    assert 'WARNING' in format_report(stats)


def test_plateau_tally_counts_only_samples_near_s_closed():
    """The tally is the evidence for the chosen value, so it must not count the whole low tail."""
    closed = np.full(120, 0.005)
    elsewhere = np.full(500, 0.005 + 10 * PLATEAU_TOL_M)

    stats = closed_width_stats(np.concatenate([closed, elsewhere]))

    assert stats.n_near_closed_width == 120


def test_nans_are_dropped_rather_than_poisoning_the_percentiles():
    """Step 4 emits NaN for undetected frames; np.percentile would return NaN for the lot."""
    series = _open_close_series()
    series[::3] = np.nan

    stats = closed_width_stats(series)

    assert np.isfinite(stats.closed_width_m)
    assert stats.n_samples == int(np.count_nonzero(np.isfinite(series)))


def test_empty_input_raises_rather_than_returning_nan():
    """A silent NaN here would propagate into gripper_calib.yaml and then into a robot command."""
    with pytest.raises(ValueError, match='no finite gripper-width samples'):
        closed_width_stats(np.full(10, np.nan))


def test_report_shows_the_percentile_table_and_marks_the_chosen_value():
    """The table is the deliverable — a human reads it to decide whether to trust the number."""
    report = format_report(closed_width_stats(_open_close_series()))

    assert 'closed width' in report
    assert f'p{DEFAULT_PERCENTILE:g}' in report
    assert 'min' in report and 'max' in report
    assert 'span' in report


def test_plateau_threshold_is_the_documented_one():
    """MIN_PLATEAU_SAMPLES is quoted in the operator-facing warning; keep them in step."""
    just_enough = np.concatenate(
        [
            np.full(MIN_PLATEAU_SAMPLES, 0.005),
            np.linspace(0.02, 0.09, 500),
        ]
    )
    just_short = np.concatenate(
        [
            np.full(MIN_PLATEAU_SAMPLES - 1, 0.005),
            np.linspace(0.02, 0.09, 500),
        ]
    )

    assert closed_width_stats(just_enough).plateau_is_convincing
    assert not closed_width_stats(just_short).plateau_is_convincing


def test_a_dense_slow_sweep_does_not_fake_a_plateau():
    """
    The reason the tally is a density comparison and not a fixed count.

    A long enough traversal drops plenty of samples into the bottom millimetre just by passing
    through it — 2000 samples over an 85 mm span leaves ~24 there, clearing MIN_PLATEAU_SAMPLES
    outright. Only comparing against the uniform expectation catches it.
    """
    stats = closed_width_stats(np.linspace(0.005, 0.090, 2000))

    assert stats.n_near_closed_width >= MIN_PLATEAU_SAMPLES, 'precondition: the absolute count passes'
    assert not stats.plateau_is_convincing, 'but the density is not'


def test_uniform_expectation_uses_the_same_band_width_the_tally_counts():
    """
    The tally is two-sided (``abs(w - closed_width) <= tol``), so the baseline must be too.

    Comparing a two-sided count against a one-sided expectation understated the baseline by
    exactly 2x, which made the density test half as strict as its own message claimed. A uniform
    sweep is the calibrating case: by construction it should score a ratio of 1.0, and it only
    does if both sides of the comparison span the same interval.
    """
    n = 4000
    stats = closed_width_stats(np.linspace(0.005, 0.090, n))

    assert stats.expected_uniform_in_band == pytest.approx(n * 2 * PLATEAU_TOL_M / stats.span_m)
    # A ramp is exactly the uniform case, so observed/expected must sit at ~1.0, not ~2.0.
    assert stats.n_near_closed_width / stats.expected_uniform_in_band == pytest.approx(1.0, abs=0.1)


def test_plateau_check_is_scale_free():
    """The same recording sampled at a higher frame rate must reach the same verdict."""
    slow = _open_close_series(dwell=200, travel=50)
    fast = _open_close_series(dwell=800, travel=200)  # 4x the frame rate, same motion

    assert closed_width_stats(slow).plateau_is_convincing
    assert closed_width_stats(fast).plateau_is_convincing


def test_a_plateau_sitting_above_a_stray_minimum_is_still_convincing():
    """
    Regression: scene d044, 16320 detections at 100% detection rate.

    Its closed dwell sat ~44.5 mm with one stray solve 3.5 mm below. Anchoring the tally on the
    raw minimum counted 8 samples beside that outlier and condemned a perfectly good recording —
    while min_is_outlier was simultaneously reporting that the minimum was not the plateau.
    """
    plateau = np.full(700, 0.0445)
    ramp = np.linspace(0.0455, 0.1323, 15000)
    stray = np.array([0.04109])

    stats = closed_width_stats(np.concatenate([plateau, ramp, stray]))

    assert stats.min_is_outlier, 'precondition: the raw minimum is a stray detection'
    assert stats.plateau_is_convincing, 'the dwell above it is what should be judged'
    assert stats.closed_width_m == pytest.approx(0.0445, abs=2e-4)

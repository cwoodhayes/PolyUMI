"""
Tests for the cross-correlation latency estimator.

This number is pasted into ``inference.yaml`` and then shifts every TF lookup on the inference
path, so the failure that matters is a confident wrong answer: an argmax over noise, or a lag
returned from the wrong end of the range. Ground truth is available for free here — delay a known
signal by a known amount and check the estimator finds it — which is what makes these worth having
despite there being no hardware in the loop.

The excitation is a chirp rather than a sine for the reason in :mod:`polyumi_ros2.latency_util`: a
narrowband excitation gives a broad correlation peak whose argmax wanders by milliseconds. These
tests therefore also pin the *conditioning* claim, not just the arithmetic.
"""

import numpy as np
import pytest
import scipy.signal as ss

from polyumi_ros2.latency_util import get_latency

#: A lag that is not a multiple of anything else in the test, so an off-by-one grid error shows.
TRUE_LAG_S = 0.087
DURATION_S = 20.0


def _chirp(f0=0.1, f1=1.5):
    """Build a linear frequency sweep over the run, as :mod:`polyumi_ros2.latency_probe` drives."""
    return lambda t: ss.chirp(np.clip(t, 0.0, DURATION_S), f0=f0, f1=f1, t1=DURATION_S, method='linear')


def _delayed(signal, lag_s=TRUE_LAG_S, target_hz=10.0, actual_hz=30.0):
    """
    Sample ``signal`` as commanded, and again delayed by ``lag_s`` as measured.

    The two series are sampled at different rates on purpose: that is the real situation (commands
    go out at control_hz, ``/fr3_gripper/joint_states`` arrives at ~30 Hz), and it exercises the
    resampling rather than letting the arrays line up by luck.
    """
    t_target = np.arange(0.0, DURATION_S, 1 / target_hz)
    t_actual = np.arange(0.0, DURATION_S, 1 / actual_hz)
    return signal(t_target), t_target, signal(t_actual - lag_s), t_actual


def test_recovers_a_known_delay():
    """The whole point: a known lag comes back within the 1 ms grid resolution."""
    latency, info = get_latency(*_delayed(_chirp()), force_positive=True)
    assert latency == pytest.approx(TRUE_LAG_S, abs=0.002)
    # A clean match should look like one: near-unit correlation at the peak.
    assert info['peak_corr'] > 0.9


def test_force_positive_does_not_return_a_lag_from_the_wrong_end():
    """
    Guard the upstream bug this port fixes.

    Upstream takes an argmax over the masked correlation and then indexes the *unmasked* lag array,
    so ``force_positive=True`` returns a large negative lag — the exact opposite of the request.
    """
    latency, _ = get_latency(*_delayed(_chirp()), force_positive=True)
    assert latency >= 0.0


def test_overlap_normalisation_beats_the_raw_argmax_on_a_narrowband_run():
    """
    The divergence from upstream earns its keep on exactly the runs that are hard to condition.

    A pure sine is what UMI drives; its correlation peak is a broad cosine, and upstream's raw
    argmax sits under a triangular overlap envelope that drags the peak toward zero lag. The arm
    mode cannot escape narrowband excitation (MoveIt's planning cadence), so this case is not
    hypothetical.
    """
    sine = lambda t: np.sin(2 * np.pi * t / 3.0)  # noqa: E731
    latency, info = get_latency(*_delayed(sine), force_positive=True)
    # Not exact — a 3 s sine cannot localise a lag well, and the wide peak says so — but the raw
    # upstream argmax lands at 0.082 on this input, further out and on the wrong side.
    assert latency == pytest.approx(TRUE_LAG_S, abs=0.010)
    assert info['peak_width_s'] > 0.2


def test_a_lag_beyond_the_search_bound_is_flagged_as_pinned():
    """
    The clamp must announce itself, because a clamped result can look like a good one.

    Caught on hardware: the gripper run returned exactly 1.000 s against a 1.0 s bound, with a
    peak_width of 138 ms that passed the sharpness check — because truncating the search window
    also truncates the measured width. Unclamped the peak was at 1.194 s and 538 ms wide, which
    would have been rejected. Without this flag the probe prints a paste-able config line for a
    number that is purely the bound.
    """
    _, clamped = get_latency(*_delayed(_chirp(), lag_s=0.5), force_positive=True, max_lag_s=0.2)
    assert clamped['pinned']
    _, ok = get_latency(*_delayed(_chirp(), lag_s=0.5), force_positive=True, max_lag_s=2.0)
    assert not ok['pinned']


def test_flat_signal_is_rejected_rather_than_answered():
    """A topic that published a constant means nothing moved — that is an error, not a 0 s lag."""
    t = np.arange(0.0, 5.0, 0.1)
    with pytest.raises(ValueError, match='zero variance'):
        get_latency(np.ones_like(t), t, np.ones_like(t), t)


def test_non_overlapping_series_are_rejected():
    """Two recordings that barely coincide in time cannot be correlated; say so."""
    t_a = np.arange(0.0, 5.0, 0.1)
    t_b = t_a + 4.999
    x = np.sin(t_a)
    with pytest.raises(ValueError, match='overlap'):
        get_latency(x, t_a, x, t_b)


def test_peak_width_separates_a_sharp_peak_from_a_sluggish_one():
    """
    The sanity check the operator is told to apply, as a number rather than an eyeballed plot.

    A narrow sweep localises the lag poorly, so its correlation peak is broad. This is the named
    failure mode for the arm mode, where MoveIt's cadence forces a slow excitation.
    """
    _, sharp = get_latency(*_delayed(_chirp(0.1, 1.5)), force_positive=True)
    _, broad = get_latency(*_delayed(_chirp(0.02, 0.1)), force_positive=True)
    assert sharp['peak_width_s'] < broad['peak_width_s']

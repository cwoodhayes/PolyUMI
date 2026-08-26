"""
Cross-correlation latency estimator: how far behind a commanded signal the measured one runs.

Ported from upstream UMI's ``umi/common/latency_util.py``, which is what UMI's
``scripts/calibrate_robot_latency.py`` and ``scripts/calibrate_gripper_latency.py`` use. Copied into
this repo rather than imported from ``external/polyumi_diffusion_policy`` so the ROS2 package owns
it: this runs under ``/usr/bin/python3`` from an installed ament package, where the submodule path
does not reliably resolve.

The method, unchanged from upstream: resample the commanded and measured series onto a common 1 kHz
grid, z-normalize them jointly, full cross-correlation, and take the lag at the peak. It needs no
model of the plant — only that the measured signal is a delayed, roughly linear echo of the
commanded one — which is why the same function serves both the arm (`/polyumi/target_poses_traj` vs the
`polyumi_tcp` TF) and the hand (`/polyumi/target_gripper` vs `/fr3_gripper/joint_states`).

Four deliberate divergences from upstream:

1. The correlation is **divided by the overlap length at each lag**. Upstream's argmax runs on the
   raw full correlation, whose values sit under a triangular envelope because fewer samples are
   compared as the lag grows — which drags the peak toward zero. UMI gets away with it by exciting
   ~10 cycles; the arm mode here cannot, since MoveIt's planning cadence forces a slow sine and
   few cycles fit in a run. Measured cost of the bias on a 6-cycle sine: 5 ms out of 87.
2. ``force_positive`` is **fixed**. Upstream does ``t_lags[np.argmax(correlation[t_lags >= 0])]``,
   which takes an argmax over the *masked* array and then indexes the *unmasked* one — so it
   returns a lag from the wrong (negative) end of the range, the exact opposite of what the flag
   asks for. Upstream never calls it with the flag set, so the bug is invisible there; we do call
   it that way, because a command cannot precede its own response.
3. Input guards, because the output is pasted into a robot config. A near-empty overlap or a
   constant (zero-variance) signal otherwise yields NaN or a garbage lag with no complaint.
4. ``info`` carries two peak-quality numbers so a bad run is caught by a printed value rather than
   by eyeballing a plot. See :func:`get_latency`.

**Accuracy is set by the excitation, not by this function.** The correlation of two narrowband
signals is itself narrowband, so its peak is broad and the argmax slides several milliseconds under
any small asymmetry. Measured against a known 87 ms lag:

    ===============================  ==========
    excitation                       error
    ===============================  ==========
    sine, 3 s period                 +7 ms
    sine, 0.94 s period (UMI's)      +1 ms
    chirp, 0.1-1.5 Hz                 0 ms
    chirp, 0.05-0.4 Hz (arm-feasible) -17 ms
    ===============================  ==========

Hence the probe drives a **chirp** rather than UMI's fixed sine — the same reason the audio time
sync in ``ingest/polyumi_ingest/preproc/time_sync.py`` moved to ``ChirpTimeSyncStep``. Where the
plant cannot be driven broadband the error is irreducible: the arm goes through MoveIt's
plan-then-execute cadence, so it is stuck in the bottom row and carries roughly +/-20 ms. That is
tolerable there and nowhere else — ``latency.arm_exec`` only sets how many leading actions get
discarded, in units of a 100 ms ``action_dt``. Read ``peak_width_s`` to see which row you are in.
"""

import numpy as np
import scipy.interpolate as si
import scipy.signal as ss

#: How close to the peak still counts as "top of the peak" when measuring its width: within
#: (1 - this) of the peak's own magnitude, i.e. 10%.
_PEAK_WIDTH_FRACTION = 0.9


def regular_sample(x: np.ndarray, t: np.ndarray, t_samples: np.ndarray) -> np.ndarray:
    """
    Linearly resample the series ``(t, x)`` onto ``t_samples``, holding the end values outside.

    Holding rather than extrapolating matches upstream, and matters at the edges: the two series
    start and end at slightly different instants, so a linear extrapolation off the end would inject
    a ramp that the correlation then happily locks onto.

    :param t: sample instants of the input series, ascending, in seconds.
    :param x: values at ``t``.
    :param t_samples: instants to resample onto, in seconds.
    :returns: ``x`` resampled at ``t_samples``.
    """
    spline = si.interp1d(x=t, y=x, bounds_error=False, fill_value=(x[0], x[-1]))
    return spline(t_samples)


def get_latency(
    x_target: np.ndarray,
    t_target: np.ndarray,
    x_actual: np.ndarray,
    t_actual: np.ndarray,
    t_start: float | None = None,
    t_end: float | None = None,
    resample_dt: float = 1 / 1000,
    force_positive: bool = False,
    max_lag_s: float = 2.0,
) -> tuple[float, dict]:
    """
    Estimate how far the measured signal lags the commanded one, in seconds.

    :param x_target: commanded values.
    :param t_target: instants of ``x_target``, in seconds on the same clock as ``t_actual``.
    :param x_actual: measured values.
    :param t_actual: instants of ``x_actual``, in seconds.
    :param t_start: start of the comparison window; defaults to the later of the two series' starts.
    :param t_end: end of the window; defaults to the earlier of the two series' ends.
    :param resample_dt: grid spacing, which is also the resolution of the result. 1 ms by default.
    :param force_positive: restrict the search to non-negative lags. Correct for any real
        command->response measurement, where a negative lag can only be a correlation artifact.
    :param max_lag_s: largest magnitude of lag to consider. Bounds the search away from the
        extremes, where the overlap normalisation divides by almost nothing.
    :returns: ``(latency_s, info)``. ``info`` holds the resampled series and the correlation curve
        for plotting, plus two quality numbers that decide whether the peak is worth believing:

        ``peak_corr``
            the overlap-normalised correlation at the winning lag. Roughly a correlation
            coefficient — near 1 means the shifted series really do coincide, near 0 means the
            argmax picked noise.
        ``peak_width_s``
            how much of the searched lag range sits within 10% of the peak — an upper bound on the
            peak's width, since it counts side lobes that clear the same threshold rather than only
            the contiguous top. A sharp peak is narrow; a wide one means the excitation was too slow
            or too small to localise the lag, and the run should be repeated rather than believed.
        ``pinned``
            True when the winning lag sits on the edge of the search window, i.e. the answer is
            the clamp rather than a peak. Always a rejection: the reported lag is meaningless and,
            because clipping the window also clips ``peak_width_s``, it can look sharp.

    :raises ValueError: if the series are mismatched, do not overlap enough to correlate, or carry
        no variation to correlate against.
    """
    if len(x_target) != len(t_target):
        raise ValueError(f'x_target/t_target length mismatch: {len(x_target)} vs {len(t_target)}')
    if len(x_actual) != len(t_actual):
        raise ValueError(f'x_actual/t_actual length mismatch: {len(x_actual)} vs {len(t_actual)}')
    if len(x_target) < 2 or len(x_actual) < 2:
        raise ValueError(f'need >= 2 samples per series, got {len(x_target)} target / {len(x_actual)} actual')

    if t_start is None:
        t_start = max(t_target[0], t_actual[0])
    if t_end is None:
        t_end = min(t_target[-1], t_actual[-1])
    n_samples = int((t_end - t_start) / resample_dt)
    # Upstream computes n_samples and correlates without checking it. Two series that barely overlap
    # (a subscription that came up late, a topic that died mid-run) then reach ss.correlate with
    # near-empty arrays and produce a lag rather than an error.
    if n_samples < 2:
        raise ValueError(
            f'target and actual overlap for only {t_end - t_start:.3f}s ({n_samples} samples at '
            f'{resample_dt}s) — the two series barely coincide in time; check both were recording'
        )

    t_samples = np.arange(n_samples) * resample_dt + t_start
    target_samples = regular_sample(x_target, t_target, t_samples)
    actual_samples = regular_sample(x_actual, t_actual, t_samples)

    # Normalize samples to zero mean unit std. Jointly, not per-series, so a difference in
    # amplitude between commanded and measured survives normalisation (a gripper that only travels
    # half as far as commanded should not be rescaled into looking like a perfect follower).
    both = np.concatenate([target_samples, actual_samples])
    mean = np.mean(both)
    std = np.std(both)
    if not np.isfinite(std) or std <= 0:
        raise ValueError(
            'the combined signal has zero variance — nothing moved, or the topic published a '
            'constant. Check the excitation actually reached the hardware'
        )
    target_samples = (target_samples - mean) / std
    actual_samples = (actual_samples - mean) / std

    # Cross correlation, divided by the number of samples actually compared at each lag.
    #
    # Upstream takes the argmax of the raw correlation, which is biased toward zero. A full
    # correlation compares fewer and fewer samples as |lag| grows, so its values sit under a
    # triangular envelope, and the argmax of (true peak x decaying envelope) lands short of the
    # true lag. On a 3 s sine it costs ~5 ms — small next to UMI's ~10-cycle excitation, but this
    # rig cannot always afford that many cycles: MoveIt's planning cadence forces the arm mode's
    # sine slow, so few cycles fit in a run and the bias grows. Dividing it out also makes the
    # value directly interpretable as a correlation coefficient.
    lags = ss.correlation_lags(len(actual_samples), len(target_samples))
    overlap = n_samples - np.abs(lags)
    correlation = ss.correlate(actual_samples, target_samples) / np.maximum(overlap, 1)
    t_lags = lags * resample_dt

    # Bound the search. Normalising by a vanishing overlap makes the extreme lags — where a couple
    # of samples happen to agree — explode, and physically the answer is a pipeline delay well
    # under a second anyway.
    mask = np.abs(t_lags) <= max_lag_s
    if force_positive:
        mask &= t_lags >= 0
    if not mask.any():
        raise ValueError(f'max_lag_s={max_lag_s} excludes every candidate lag')
    # NB the double index. Upstream indexes the unmasked t_lags with an index into the masked
    # correlation, which lands at the wrong end of the lag range entirely.
    candidates = np.flatnonzero(mask)
    peak_idx = int(candidates[np.argmax(correlation[mask])])
    latency = float(t_lags[peak_idx])

    peak = correlation[peak_idx]
    peak_corr = float(peak)
    # A COUNT of every sample within 10% of the peak, not the width of the contiguous run around
    # it — so a side lobe that also clears the threshold is counted in. Deliberate: the metric only
    # ever gates a rejection (peak_width_s > MAX_PEAK_WIDTH_S), so over-counting can only make the
    # probe ask for a re-run, never wave a bad number through. It rarely fires anyway on the
    # shipped excitation — the arm chirp's slowest component has a 20 s period and its fastest a
    # 2.5 s one, so its side lobes fall outside the +/-2 s search window entirely. Only worth
    # replacing with a contiguous run if a future excitation puts lobes inside the window and the
    # re-run requests get annoying.
    #
    # Written against the peak's magnitude rather than as `peak * _PEAK_WIDTH_FRACTION`, which
    # inverts the comparison for a negative peak (0.9x a negative number is LARGER than it) and
    # would report a width of one sample for a correlation that never matched at all. Callers
    # reject those on peak_corr first, so this is belt-and-braces.
    threshold = peak - abs(peak) * (1 - _PEAK_WIDTH_FRACTION)
    peak_width_s = float(np.count_nonzero(correlation[mask] >= threshold) * resample_dt)
    # A winner sitting on the edge of the search window is not a peak, it is the clamp: the true
    # maximum is at or beyond the bound and we cropped it. Callers must reject these rather than
    # report the bound as a measurement. Worth flagging explicitly because truncating the window
    # also truncates peak_width_s, so a clamped result can look deceptively sharp — a real run
    # produced 1.000 s at width 138 ms whose unclamped peak was 1.194 s at width 538 ms, i.e. it
    # would have failed a width check but sailed through the clamped one.
    pinned = bool(peak_idx in (candidates[0], candidates[-1]))

    info = {
        't_samples': t_samples,
        'x_target': target_samples,
        'x_actual': actual_samples,
        'correlation': correlation,
        'lags': t_lags,
        'peak_corr': peak_corr,
        'peak_width_s': peak_width_s,
        'pinned': pinned,
        'max_lag_s': max_lag_s,
    }
    return latency, info

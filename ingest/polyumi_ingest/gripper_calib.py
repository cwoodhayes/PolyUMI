"""
Deriving the closed-gripper ArUco width (``S_closed``) from a recorded open/close session.

This is PolyUMI's answer to UMI's ``scripts/calibrate_gripper_range.py``, which takes a plain
``np.nanmin`` / ``np.nanmax`` over every frame of a calibration video. That statistic is brittle in
a way that is invisible in its output: two bad PnP solves set the whole range, and a single number
cannot distinguish "the gripper sat closed for 200 frames here" from "one frame reconstructed
badly". Both are just a small float.

So the reduction reports a *shape* instead of a number. A genuine closed plateau shows up two ways:
the low percentiles cluster tightly together, and many samples sit within a millimetre of the
minimum. An outlier shows up as a minimum detached from p1 with a sample count of one or two. The
caller picks from that, and :func:`format_report` renders it for a human to eyeball.

The measurement itself is ingest step 4's (``aruco_step.py``): ``right_tvec.x - left_tvec.x``, the
finger-tag centre separation in metres, depth-gated to detections landing at ``nominal_z``. Feed
this ``raw_widths_m`` — the actual per-frame detections — and never ``width_m``, which has been
resampled onto the GoPro grid with hold-at-edges extrapolation and would drag the extremes toward
whatever the series happened to start and end on.

What ``S_closed`` is *for* is written up in docs/franka-inference-bringup.md: it is the tag
separation with the fingers touching, which the DP exporter subtracts so that exported widths are
opening-from-closed (UMI's convention), and which the FR3 side pairs with its own closed aperture.
"""

from __future__ import annotations

import dataclasses

import numpy as np

#: Percentiles reported alongside the raw min, low to high. The point is to see where the low tail
#: stops moving: if p0.5 through p5 are all within a millimetre, that flat region is the closed
#: plateau. If the minimum sits well below p1, it is an outlier rather than a measurement.
REPORT_PERCENTILES = (0.5, 1.0, 2.0, 5.0)

#: Which of the above is taken as ``S_closed`` by default. p1 keeps a handful of bad detections out
#: of a number that lands in a robot command, while staying on the plateau for any recording where
#: the gripper is actually shut for a reasonable fraction of the time.
DEFAULT_PERCENTILE = 1.0

#: Half-width of the band counted as "at the plateau". Loose relative to ArUco's real precision on
#: a 16 mm tag, deliberately: the question it answers is "was the gripper parked here", not "how
#: repeatable is the detector".
PLATEAU_TOL_M = 0.001

#: Below this many samples on the plateau, the minimum is probably noise rather than a dwell.
MIN_PLATEAU_SAMPLES = 20

#: ...but a raw count is not enough on its own. A long, slow sweep that never stops at the closed
#: position still drops plenty of samples into the bottom millimetre purely by passing through it —
#: a 2000-sample ramp over an 85 mm span leaves ~24 there, clearing any fixed threshold. So the
#: tally is also compared against the density a *uniform* traversal of the same span would produce,
#: and has to beat it by this factor. That makes the check scale-free: independent of frame rate,
#: recording length, and how many cycles the operator did.
PLATEAU_DENSITY_FACTOR = 3.0


@dataclasses.dataclass(frozen=True)
class ClosedWidthStats:
    """The shape of a width series' low tail, from which ``S_closed`` is read off."""

    n_samples: int
    min_m: float
    max_m: float
    percentiles_m: dict[float, float]
    #: Samples within :data:`PLATEAU_TOL_M` of ``s_closed_m`` — the evidence that it is a dwell.
    #:
    #: Anchored on the chosen value, NOT on ``min_m``. Anchoring on the minimum measures the
    #: neighbourhood of whatever the worst detection was: on a real recording whose plateau sat
    #: 3.5 mm above a stray solve, it counted 8 samples beside the outlier and declared a perfectly
    #: good session unusable, while ``min_is_outlier`` was simultaneously reporting that the
    #: minimum was not the plateau.
    n_near_s_closed: int
    #: The value to use, i.e. ``percentiles_m[DEFAULT_PERCENTILE]``.
    s_closed_m: float

    @property
    def span_m(self) -> float:
        """Full observed travel, i.e. the width range the policy saw in training."""
        return self.max_m - self.min_m

    @property
    def expected_uniform_in_band(self) -> float:
        """How many samples would fall in a PLATEAU_TOL_M band if the width swept uniformly."""
        if self.span_m <= PLATEAU_TOL_M:
            # Everything is within tolerance of everything: there is no travel to sweep, so the
            # comparison is meaningless and the absolute count is the only test that applies.
            return 0.0
        return self.n_samples * PLATEAU_TOL_M / self.span_m

    @property
    def plateau_is_convincing(self) -> bool:
        """Whether S_closed sits in a dwell rather than on a ramp the width merely passed through."""
        dense_enough = self.n_near_s_closed >= PLATEAU_DENSITY_FACTOR * self.expected_uniform_in_band
        return self.n_near_s_closed >= MIN_PLATEAU_SAMPLES and dense_enough

    @property
    def min_is_outlier(self) -> bool:
        """Whether the raw minimum sits detached from the low percentiles."""
        return (self.percentiles_m[DEFAULT_PERCENTILE] - self.min_m) > PLATEAU_TOL_M


def closed_width_stats(widths_m: np.ndarray) -> ClosedWidthStats:
    """
    Summarise a pooled ArUco width series' low tail.

    :param widths_m: per-frame finger-tag separations in metres; NaNs are dropped.
    :returns: the percentile table, plateau tally and resulting ``S_closed``.
    :raises ValueError: if no finite samples remain, since every downstream number would be NaN.
    """
    finite = np.asarray(widths_m, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError('no finite gripper-width samples; did step 4 detect any finger tags?')

    percentiles = {
        float(p): float(np.percentile(finite, p)) for p in REPORT_PERCENTILES
    }
    minimum = float(finite.min())
    s_closed = percentiles[DEFAULT_PERCENTILE]
    return ClosedWidthStats(
        n_samples=int(finite.size),
        min_m=minimum,
        max_m=float(finite.max()),
        percentiles_m=percentiles,
        n_near_s_closed=int(np.count_nonzero(np.abs(finite - s_closed) <= PLATEAU_TOL_M)),
        s_closed_m=s_closed,
    )


def format_report(stats: ClosedWidthStats) -> str:
    """Render the stats for a human, in millimetres, with the judgement left visible."""
    lines = [
        f'samples          {stats.n_samples}',
        f'min              {stats.min_m * 1000:.2f} mm   (UMI takes this verbatim)',
    ]
    lines += [
        f'p{p:<15g} {stats.percentiles_m[p] * 1000:.2f} mm'
        + ('   <- S_closed' if p == DEFAULT_PERCENTILE else '')
        for p in REPORT_PERCENTILES
    ]
    lines += [
        f'max              {stats.max_m * 1000:.2f} mm',
        f'span             {stats.span_m * 1000:.2f} mm   (width range seen in training)',
        f'within {PLATEAU_TOL_M * 1000:.0f}mm of S_closed  {stats.n_near_s_closed} samples '
        f'({stats.expected_uniform_in_band:.0f} expected if it were only passing through)',
    ]
    if not stats.plateau_is_convincing:
        lines.append(
            f'WARNING: only {stats.n_near_s_closed} samples near S_closed, against '
            f'{stats.expected_uniform_in_band:.0f} expected from a uniform sweep '
            f'(want >= {MIN_PLATEAU_SAMPLES} and >= {PLATEAU_DENSITY_FACTOR:g}x that). The gripper '
            'looks like it passed through the closed position rather than resting at it — hold it '
            'shut for a few seconds per cycle and re-record.'
        )
    if stats.min_is_outlier:
        lines.append(
            f'NOTE: the raw min is {(stats.percentiles_m[DEFAULT_PERCENTILE] - stats.min_m) * 1000:.2f} mm '
            'below p1, so it is a stray detection rather than the plateau. This is exactly the case '
            "UMI's plain nanmin would have taken at face value."
        )
    return '\n'.join(lines)

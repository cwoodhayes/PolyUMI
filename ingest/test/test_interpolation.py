"""Tests for SE(3) trajectory gap-filling (interpolation.py)."""

import numpy as np
from scipy.spatial.transform import Rotation

from polyumi_ingest.interpolation import gap_runs, interpolate_se3_gaps


def _poses(rots: Rotation, trans: np.ndarray) -> np.ndarray:
    """Pack rotations + translations into an (N,7) [xyz, quat] array."""
    return np.concatenate([np.atleast_2d(trans), np.atleast_2d(rots.as_quat())], axis=1)


def test_gap_runs_classifies_lead_trail_interior() -> None:
    """Leading, trailing, and interior NaN runs are separated correctly."""
    valid = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0], dtype=bool)
    interior, lead, trail = gap_runs(valid)
    assert lead == 2
    assert trail == 1
    assert interior == [(4, 3)]


def test_gap_runs_all_valid_and_all_nan() -> None:
    """Degenerate masks report no interior gaps."""
    assert gap_runs(np.ones(5, dtype=bool)) == ([], 0, 0)
    assert gap_runs(np.zeros(5, dtype=bool)) == ([], 5, 5)


def test_interpolate_recovers_a_held_out_pose() -> None:
    """A single missing frame between two knowns is recovered near the true midpoint pose."""
    r0 = Rotation.identity()
    r2 = Rotation.from_euler('z', 90, degrees=True)
    r1_true = Rotation.from_euler('z', 45, degrees=True)  # slerp midpoint
    poses = _poses(
        Rotation.concatenate([r0, r1_true, r2]),
        np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6]], dtype=float),
    )
    holed = poses.copy()
    holed[1] = np.nan

    filled, mask = interpolate_se3_gaps(holed, max_gap_frames=5)

    assert mask.tolist() == [False, True, False]
    np.testing.assert_allclose(filled[1, :3], [1, 2, 3], atol=1e-9)  # linear midpoint
    rot_err = (Rotation.from_quat(filled[1, 3:]).inv() * r1_true).magnitude()
    assert rot_err < 1e-9


def test_over_long_gap_is_left_nan() -> None:
    """Gaps longer than max_gap_frames stay NaN and are not marked filled."""
    poses = _poses(Rotation.identity(6), np.tile(np.arange(6)[:, None], (1, 3)).astype(float))
    poses[2:5] = np.nan  # interior gap of length 3

    filled, mask = interpolate_se3_gaps(poses, max_gap_frames=2)

    assert not mask.any()
    assert np.isnan(filled[2:5]).all()


def test_leading_and_trailing_nan_untouched() -> None:
    """Runs without a bracket on one side cannot be interpolated."""
    poses = _poses(Rotation.identity(5), np.tile(np.arange(5)[:, None], (1, 3)).astype(float))
    poses[0] = np.nan  # leading
    poses[4] = np.nan  # trailing

    filled, mask = interpolate_se3_gaps(poses, max_gap_frames=10)

    assert not mask.any()
    assert np.isnan(filled[0]).all()
    assert np.isnan(filled[4]).all()

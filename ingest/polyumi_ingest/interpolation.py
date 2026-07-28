"""SE(3) trajectory gap-filling for sparse pose sources (e.g. SLAM tracking loss)."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def gap_runs(valid: np.ndarray) -> tuple[list[tuple[int, int]], int, int]:
    """
    Classify the NaN runs of a boolean validity mask.

    A pose trajectory on a uniform frame grid has three kinds of missing data:
    a **leading** run (before tracking initialises), a **trailing** run (after it is
    lost for good), and **interior** runs bracketed by a valid pose on both sides.
    Only interior runs can be interpolated — the others have no bracket on one side.

    Args:
        valid: (N,) bool, True where the frame has a pose.

    Returns:
        (interior_gaps, lead_len, trail_len) where interior_gaps is a list of
        ``(start_index, length)`` for each interior NaN run.

    """
    valid = np.asarray(valid, dtype=bool)
    n = len(valid)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not valid[i]:
            j = i
            while j < n and not valid[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    lead = runs[0][1] if runs and runs[0][0] == 0 else 0
    trail = runs[-1][1] if runs and runs[-1][0] + runs[-1][1] == n else 0
    interior = [(s, length) for (s, length) in runs if s > 0 and s + length < n]
    return interior, lead, trail


def interpolate_se3_gaps(poses: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill short interior gaps of an SE(3) trajectory in place-safe copy.

    Rotation is interpolated with Slerp (the constant-angular-velocity geodesic on SO(3));
    translation is interpolated linearly. The two are decoupled — componentwise quaternion
    interpolation would be wrong, and full SE(3) screw interpolation is not worth the extra
    machinery at the short gap lengths this targets. Only *interior* gaps no longer than
    ``max_gap_frames`` are filled; leading/trailing NaN and over-long gaps are left as NaN
    (downstream consumers trim to the valid span).

    Args:
        poses: (N, 7) [x y z qx qy qz qw]; NaN rows mark missing frames (e.g. SLAM loss).
        max_gap_frames: longest interior gap (in frames) that will be filled.

    Returns:
        (filled_poses, filled_mask): a copy of ``poses`` with fillable gaps interpolated,
        and a (N,) bool mask marking the frames this function wrote.

    """
    poses = np.asarray(poses, dtype=np.float64).copy()
    filled = np.zeros(len(poses), dtype=bool)
    valid = ~np.isnan(poses).any(axis=1)
    interior, _, _ = gap_runs(valid)
    for s, length in interior:
        if length > max_gap_frames:
            continue
        lo, hi = s - 1, s + length  # bracketing valid frame indices
        fracs = np.arange(1, length + 1) / (length + 1)
        # translation: linear between the bracketing poses
        poses[s : s + length, :3] = (1 - fracs)[:, None] * poses[lo, :3] + fracs[:, None] * poses[hi, :3]
        # rotation: slerp along the SO(3) geodesic between the bracketing orientations
        slerp = Slerp([0.0, 1.0], Rotation.from_quat(poses[[lo, hi], 3:]))
        poses[s : s + length, 3:] = slerp(fracs).as_quat()
        filled[s : s + length] = True
    return poses, filled

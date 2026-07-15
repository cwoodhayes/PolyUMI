"""Shared helpers for aligning PolyUMI's sensor streams onto a common time base."""

import numpy as np


def nearest_idx(sorted_ts: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Index of the nearest value in ascending ``sorted_ts`` for each ``query`` time.

    Nearest-neighbour rather than interpolation: the callers resample poses and gripper
    widths, and interpolating a quaternion componentwise is wrong while slerping every
    sample is not worth it at the rates involved (sources run at 60-120 Hz, targets at 10).

    Args:
        sorted_ts: (N,) source timestamps, ascending.
        query: (M,) times to look up. Values outside the source range clamp to the ends.

    Returns:
        (M,) int indices into ``sorted_ts``.

    Raises:
        RuntimeError: if fewer than 2 source timestamps are given, which leaves nothing
            meaningful to resample from.

    """
    if len(sorted_ts) < 2:
        raise RuntimeError(f'Need at least 2 timestamps to resample, got {len(sorted_ts)}')
    idx = np.searchsorted(sorted_ts, query)
    idx = np.clip(idx, 1, len(sorted_ts) - 1)
    closer_left = (query - sorted_ts[idx - 1]) <= (sorted_ts[idx] - query)
    idx = idx - closer_left
    return np.clip(idx, 0, len(sorted_ts) - 1)

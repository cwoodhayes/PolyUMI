"""Shared helpers for aligning PolyUMI's sensor streams onto a common time base."""

import numpy as np
import zarr

from polyumi_ingest.pzarr.store import arr, grp


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


def gopro_ts_in_finger_clock(ep: zarr.Group, *, require_offset: bool) -> np.ndarray:
    """
    GoPro frame timestamps shifted into the finger (= OptiTrack) clock domain.

    The GoPro and the Pi stamp against unrelated epochs. ``annotations/time_sync`` (step 1)
    measures the difference by matched-filtering the sync chirp against both recordings, and
    subtracting it puts GoPro times in the domain ``timestamps/finger*`` lives in.

    Args:
        ep: episode group.
        require_offset: whether a missing ``annotations/time_sync`` is fatal. Callers that
            merely resample a slowly-varying signal tolerate the unshifted grid; callers that
            slice audio sample-exactly do not, because a silently unshifted grid is wrong by
            the full inter-device offset — seconds, not milliseconds — and nothing downstream
            can detect it.

    Returns:
        (N,) GoPro frame times in the finger clock.

    Raises:
        RuntimeError: if ``require_offset`` and the scene has no chirp time-sync annotation.

    """
    ts = np.asarray(arr(ep, 'timestamps/gopro')[:], dtype=np.float64)
    offset_attr = None
    if 'annotations/time_sync' in ep:
        offset_attr = grp(ep, 'annotations/time_sync').attrs.get('gopro_to_finger_offset_s')
    if offset_attr is None:
        if require_offset:
            raise RuntimeError(
                'no annotations/time_sync (or it has no gopro_to_finger_offset_s) — run '
                'preprocessing step 1 (chirp-time-sync) first. Without it the GoPro and finger '
                'clocks are unrelated epochs, so any sample-exact alignment against finger audio '
                'would be silently wrong.'
            )
        return ts
    return ts - float(offset_attr)

"""
Structure of the missing data in a sparse pose trajectory.

Once held gap *filling* too (Slerp+linear over short interior gaps). That was removed with
pzarr v4: nothing in the pipeline invents a pose any more, so what is left is the analysis
that classifies where the holes are — used to reason about SLAM coverage and, at export, to
cut a trajectory into contiguous usable segments.
"""

import numpy as np


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

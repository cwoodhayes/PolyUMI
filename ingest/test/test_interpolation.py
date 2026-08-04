"""Tests for missing-data classification in sparse pose trajectories (interpolation.py)."""

import numpy as np

from polyumi_ingest.interpolation import gap_runs


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

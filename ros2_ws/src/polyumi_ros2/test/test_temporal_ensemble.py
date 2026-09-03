"""
Tests for temporal ensembling of overlapping action chunks.

The failure modes here are all silent. A blend that quietly extrapolates past a chunk's horizon
still returns plausible numbers; so does one that averages two quaternions naming the same
rotation with opposite signs, right up until the arm rolls through an orientation nobody asked
for. And a blend that survives an episode reset drags the first chunk of the new episode back
towards where the arm was in the old one. These pin each of those, plus the two degenerate cases
that must reduce exactly to today's behaviour: disabled, and the first chunk of an episode.
"""

import numpy as np
import pytest

from polyumi_ros2.temporal_ensemble import TemporalEnsembler

ACTION_DT = 0.1
TAU_S = 0.3
GRIP_COL = 7


def chunk(n: int, x0: float, dx: float = 0.0, grip: float = 0.05) -> np.ndarray:
    """Build a chunk whose x ramps from x0 by dx per step, with identity rotation."""
    actions = np.zeros((n, 8))
    actions[:, 0] = x0 + dx * np.arange(n)
    actions[:, 6] = 1.0  # qw
    actions[:, 7] = grip
    return actions


def test_disabled_returns_the_chunk_untouched():
    """A tau of 0 is the off switch: the node must behave exactly as it did before this existed."""
    ens = TemporalEnsembler(tau_s=0.0, action_dt=ACTION_DT)
    first = chunk(4, x0=0.0)
    ens.blend(t_obs=10.0, actions=first)
    second = chunk(4, x0=1.0)

    assert np.array_equal(ens.blend(t_obs=10.3, actions=second), second)


def test_first_chunk_of_an_episode_is_returned_unchanged():
    """With nothing to blend against, blending must be a no-op rather than a scaled-down chunk."""
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    first = chunk(4, x0=0.0, dx=0.01)

    assert ens.blend(t_obs=10.0, actions=first) == pytest.approx(first)


def test_blend_pulls_the_new_chunk_towards_the_previous_one():
    """The whole point: a chunk that disagrees with its predecessor is not executed at face value."""
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    ens.blend(t_obs=10.0, actions=chunk(8, x0=0.0))
    jumped = chunk(8, x0=1.0)

    blended = ens.blend(t_obs=10.3, actions=jumped)

    # The old chunk said 0.0 everywhere it still covers, so the blend must land strictly between.
    covered = blended[:5, 0]
    assert np.all(covered > 0.0)
    assert np.all(covered < 1.0)


def test_weighting_is_towards_the_newest_chunk():
    """Recency has to win, or the arm tracks a stale prediction with fresh timestamps on it."""
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    ens.blend(t_obs=10.0, actions=chunk(8, x0=0.0))

    blended = ens.blend(t_obs=10.3, actions=chunk(8, x0=1.0))

    # exp(-0.3/0.3) = 0.368 for the old chunk against 1.0 for the new, so the new one holds the
    # majority of the weight: the blend sits above the midpoint.
    assert blended[0, 0] > 0.5


def test_a_larger_tau_blends_harder():
    """The tau knob trades smoothness against reactivity, so it has to move the result."""
    gentle = TemporalEnsembler(tau_s=0.1, action_dt=ACTION_DT)
    heavy = TemporalEnsembler(tau_s=1.0, action_dt=ACTION_DT)
    for ens in (gentle, heavy):
        ens.blend(t_obs=10.0, actions=chunk(8, x0=0.0))

    gentle_x = gentle.blend(t_obs=10.3, actions=chunk(8, x0=1.0))[0, 0]
    heavy_x = heavy.blend(t_obs=10.3, actions=chunk(8, x0=1.0))[0, 0]

    # Heavier smoothing keeps more of the old chunk, so it stays further from the new chunk's 1.0.
    assert heavy_x < gentle_x


def test_no_extrapolation_past_a_previous_chunks_horizon():
    """
    Past its last waypoint an old chunk has no opinion and must contribute nothing.

    np.interp holds the endpoint instead of refusing, which would let a chunk that ended a second
    ago keep pulling on the far tail of every chunk after it.
    """
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    # Covers t = 10.0 .. 10.3 only.
    ens.blend(t_obs=10.0, actions=chunk(4, x0=0.0))
    new = chunk(8, x0=1.0)

    blended = ens.blend(t_obs=10.2, actions=new)

    # t = 10.2, 10.3 overlap the old chunk and are pulled down; 10.4 onwards are past its end and
    # must be exactly what the policy asked for.
    assert blended[0, 0] < 1.0
    assert blended[1, 0] < 1.0
    assert blended[2:, 0] == pytest.approx(new[2:, 0])


def test_reset_forgets_the_previous_episode():
    """A chunk from the last episode describes an arm that has since jumped back to a start pose."""
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    ens.blend(t_obs=10.0, actions=chunk(8, x0=0.0))
    ens.reset()
    fresh = chunk(8, x0=1.0)

    assert ens.blend(t_obs=10.3, actions=fresh) == pytest.approx(fresh)


def test_opposite_sign_quaternions_reinforce_instead_of_cancelling():
    """
    A quaternion and its negation are the same rotation; averaging them naively cancels out.

    The model is free to emit either sign, so this is not a contrived input — and the failure is
    invisible until the normalisation blows a near-zero vector up into an arbitrary orientation.
    """
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    first = chunk(8, x0=0.0)
    first[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    ens.blend(t_obs=10.0, actions=first)

    second = chunk(8, x0=0.0)
    second[:, 3:7] = [0.0, 0.0, 0.0, -1.0]  # identical rotation, opposite sign
    blended = ens.blend(t_obs=10.3, actions=second)

    # Whatever sign it settles on, the result must still be the identity rotation and unit norm.
    assert np.abs(blended[:, 6]) == pytest.approx(np.ones(8))
    assert np.linalg.norm(blended[:, 3:7], axis=1) == pytest.approx(np.ones(8))


def test_blended_quaternions_stay_unit_length():
    """Downstream treats these as rotations; a non-unit quaternion is a silently skewed pose."""
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    first = chunk(8, x0=0.0)
    first[:, 3:7] = [0.0, 0.0, np.sin(0.2), np.cos(0.2)]
    ens.blend(t_obs=10.0, actions=first)

    second = chunk(8, x0=0.0)
    second[:, 3:7] = [0.0, 0.0, np.sin(0.5), np.cos(0.5)]
    blended = ens.blend(t_obs=10.3, actions=second)

    assert np.linalg.norm(blended[:, 3:7], axis=1) == pytest.approx(np.ones(8))


def test_gripper_width_is_blended_too():
    """The width rides the same weights as the pose; pinned so a change to that is deliberate."""
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    ens.blend(t_obs=10.0, actions=chunk(8, x0=0.0, grip=0.00))

    blended = ens.blend(t_obs=10.3, actions=chunk(8, x0=0.0, grip=0.08))

    assert 0.0 < blended[0, GRIP_COL] < 0.08


def test_output_grid_is_the_new_chunks_own_timeline():
    """
    Blending must not change the length or the meaning of index i.

    Everything downstream — the stale-drop count, first_index, the anchor stamp — assumes action i
    targets t_obs + i * action_dt of the chunk it came back with.
    """
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    ens.blend(t_obs=10.0, actions=chunk(12, x0=0.0))
    new = chunk(16, x0=1.0)

    assert ens.blend(t_obs=10.3, actions=new).shape == new.shape


def test_a_steady_stream_of_agreeing_chunks_is_a_fixed_point():
    """Pin that a policy which stops changing its mind is passed through without drift."""
    ens = TemporalEnsembler(tau_s=TAU_S, action_dt=ACTION_DT)
    # Every chunk describes the same absolute ramp x(T) = 0.1 * (T - 10.0), just seen from a later
    # vantage point, so each one agrees with its predecessors everywhere they overlap.
    last = None
    for k in range(6):
        t_obs = 10.0 + 0.3 * k
        last = ens.blend(t_obs=t_obs, actions=chunk(16, x0=0.1 * (t_obs - 10.0), dx=0.01))

    assert last == pytest.approx(chunk(16, x0=0.1 * 1.5, dx=0.01))

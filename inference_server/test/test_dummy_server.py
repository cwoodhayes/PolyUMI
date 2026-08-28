"""
Tests for the dummy server's oscillator and its HOME_POSE validation.

The dummy is the only thing exercising the ROS-side action path without a GPU or a checkpoint, so
its output shape matters: if the gripper channel is constant, a broken gripper route looks identical
to a working one.

What it *refuses* is tested in test_client_server.py, against the shared app rather than against
this server in particular -- that agreement is the whole point of the app being shared.
"""

import os
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from polyumi_inference import Observation
from polyumi_inference.backends.sine import (
    GRIPPER_OSCILLATION_AMPLITUDE_M,
    OSCILLATION_AMPLITUDE_M,
    OSCILLATION_PERIOD_STEPS,
)
from polyumi_inference.dummy_server import app

HOME_GRIPPER = 0.05
HOME_X = 0.56
HOME_POSE = f'{HOME_X} 0.13 0.25 -1 0 0 0 {HOME_GRIPPER}'


def _post(client, n_action_steps: int = 8, n_obs_steps: int = 2):
    """POST a structurally valid frame the way policy_client_node does."""
    obs = Observation(
        channels={
            'camera0_rgb': np.full((n_obs_steps, 8, 8, 3), 128, dtype=np.uint8),
            'agent_pos': np.array([[0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.04]] * n_obs_steps),
        },
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )
    return client.post(
        '/predict_cartesian/',
        content=obs.to_frame(),
        headers={'Content-Type': 'application/octet-stream'},
    )


@pytest.fixture
def client():
    """Build a TestClient with a known HOME_POSE, so the oscillation centre is predictable."""
    with patch.dict(os.environ, {'HOME_POSE': HOME_POSE}), TestClient(app) as test_client:
        yield test_client


def test_gripper_oscillates(client):
    """
    The gripper channel actually varies -- the point of the whole exercise.

    A constant here would make a dropped or mis-routed gripper command indistinguishable from a
    working one during bringup.
    """
    widths = []
    for _ in range(4):
        widths.extend(a[7] for a in _post(client).json()['actions'])

    assert len(set(widths)) > 1
    assert min(widths) < HOME_GRIPPER < max(widths)


def test_gripper_stays_in_a_plausible_range(client):
    """Widths stay non-negative and within the amplitude of the home width."""
    widths = []
    for _ in range(4):
        widths.extend(a[7] for a in _post(client).json()['actions'])

    assert min(widths) >= 0.0
    assert max(widths) <= HOME_GRIPPER + GRIPPER_OSCILLATION_AMPLITUDE_M + 1e-9


def test_gripper_is_a_quarter_period_out_of_phase_with_x(client):
    """
    X and the gripper share a frequency but not a phase -- deliberately.

    X is a sine and the gripper a cosine about their home values, so where X crosses its centre the
    gripper is at an extreme. That is what makes a routing bug (X wired into the gripper) visible on
    inspection instead of merely plausible. Checking sin^2 + cos^2 == 1 pins the relationship without
    depending on which sample lands where.
    """
    # One full period, so the assertion covers every phase rather than a lucky few. The backend caps
    # each chunk at its own horizon, so this takes several calls.
    actions = []
    while len(actions) < OSCILLATION_PERIOD_STEPS:
        actions.extend(_post(client, n_action_steps=OSCILLATION_PERIOD_STEPS).json()['actions'])

    for action in actions:
        sin_component = (action[0] - HOME_X) / OSCILLATION_AMPLITUDE_M
        cos_component = (action[7] - HOME_GRIPPER) / GRIPPER_OSCILLATION_AMPLITUDE_M
        assert sin_component**2 + cos_component**2 == pytest.approx(1.0, abs=1e-6)


def test_phase_advances_across_calls(client):
    """Consecutive calls continue the waveform rather than restarting from the same phase."""
    first = _post(client).json()['actions']
    second = _post(client).json()['actions']

    assert first[0][7] != pytest.approx(second[0][7])


def test_chunk_is_capped_at_the_model_horizon(client):
    """
    Asking for more steps than a checkpoint emits gets the checkpoint's horizon, not padding.

    The client's stale-action arithmetic is tuned against the real chunk length, so a dummy that
    returned whatever was asked for would make bringup disagree with production.
    """
    assert len(_post(client, n_action_steps=64).json()['actions']) == 8


@pytest.mark.parametrize('bad_gripper', ['0.4', '0', '-0.02', '5'])
def test_home_pose_rejects_implausible_gripper_width(bad_gripper):
    """
    An out-of-range home width fails at startup instead of being commanded at the hand.

    The shipped default really was 0.4 (400 mm, ~5x the Franka Hand's stroke) and went unnoticed only
    because the width was being dropped downstream. Once it is routed, that is a goal the gripper
    aborts on every tick.
    """
    home = f'0.56 0.13 0.25 -1 0 0 0 {bad_gripper}'
    with patch.dict(os.environ, {'HOME_POSE': home}):
        with pytest.raises(ValueError, match='gripper width'):
            with TestClient(app):
                pass


def test_home_pose_rejects_wrong_length():
    """A HOME_POSE with the wrong number of values is rejected with a clear message."""
    with patch.dict(os.environ, {'HOME_POSE': '0.5 0.1 0.2'}):
        with pytest.raises(ValueError, match='must have 8 values'):
            with TestClient(app):
                pass

"""
Tests for the dummy inference server's oscillator and HOME_POSE validation.

The dummy is the only thing exercising the ROS-side action path without a GPU or a checkpoint,
so its output shape matters: if the gripper channel is constant, a broken gripper route looks
identical to a working one.
"""

import os
import pathlib
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from inference_server.dummy_server import (
    GRIPPER_OSCILLATION_AMPLITUDE_M,
    OSCILLATION_AMPLITUDE_M,
    OSCILLATION_PERIOD_STEPS,
    app,
)
from inference_server.obs_wire import WireFormatError, pack_observation, unpack_observation

HOME_GRIPPER = 0.05
HOME_POSE = f'0.56 0.13 0.25 -1 0 0 0 {HOME_GRIPPER}'


def _request_body(n_action_steps: int = 8, n_obs_steps: int = 2) -> bytes:
    """Build a structurally valid /predict_cartesian/ frame."""
    # uint8, as the client sends and as the dataset stores camera0_rgb.
    return pack_observation(
        {
            'camera0_rgb': np.full((n_obs_steps, 8, 8, 3), 128, dtype=np.uint8),
            'agent_pos': np.array([[0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.04]] * n_obs_steps),
        },
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )


def _post(client, body: bytes):
    """POST a packed frame the way policy_client_node does."""
    return client.post('/predict_cartesian/', content=body, headers={'Content-Type': 'application/octet-stream'})


@pytest.fixture
def client():
    """Build a TestClient with a known HOME_POSE, so the oscillation centre is predictable."""
    with patch.dict(os.environ, {'HOME_POSE': HOME_POSE}), TestClient(app) as test_client:
        yield test_client


def test_gripper_oscillates(client):
    """
    The gripper channel actually varies — the point of the whole exercise.

    A constant here would make a dropped or mis-routed gripper command indistinguishable from a
    working one during bringup.
    """
    widths = []
    for _ in range(4):
        actions = _post(client, _request_body()).json()['actions']
        widths.extend(a[7] for a in actions)

    assert len(set(widths)) > 1
    assert min(widths) < HOME_GRIPPER < max(widths)


def test_gripper_stays_in_a_plausible_range(client):
    """Widths stay non-negative and within the amplitude of the home width."""
    widths = []
    for _ in range(4):
        actions = _post(client, _request_body()).json()['actions']
        widths.extend(a[7] for a in actions)

    assert min(widths) >= 0.0
    assert max(widths) <= HOME_GRIPPER + GRIPPER_OSCILLATION_AMPLITUDE_M + 1e-9


def test_gripper_is_a_quarter_period_out_of_phase_with_x(client):
    """
    X and the gripper share a frequency but not a phase — deliberately.

    X is a sine and the gripper a cosine about their home values, so where X crosses its centre
    the gripper is at an extreme. That is what makes a routing bug (X wired into the gripper)
    visible on inspection instead of merely plausible. Checking sin^2 + cos^2 == 1 pins the
    relationship without depending on which sample lands where.
    """
    # One full period, so the assertion covers every phase rather than a lucky few.
    actions = _post(client, _request_body(n_action_steps=OSCILLATION_PERIOD_STEPS)).json()['actions']

    for action in actions:
        sin_component = (action[0] - 0.56) / OSCILLATION_AMPLITUDE_M
        cos_component = (action[7] - HOME_GRIPPER) / GRIPPER_OSCILLATION_AMPLITUDE_M
        assert sin_component**2 + cos_component**2 == pytest.approx(1.0, abs=1e-6)


def test_phase_advances_across_calls(client):
    """Consecutive calls continue the waveform rather than restarting from the same phase."""
    first = _post(client, _request_body()).json()['actions']
    second = _post(client, _request_body()).json()['actions']

    assert first[0][7] != pytest.approx(second[0][7])


@pytest.mark.parametrize('bad_gripper', ['0.4', '0', '-0.02', '5'])
def test_home_pose_rejects_implausible_gripper_width(bad_gripper):
    """
    An out-of-range home width fails at startup instead of being commanded at the hand.

    The shipped default really was 0.4 (400 mm, ~5x the Franka Hand's stroke) and went unnoticed
    only because the width was being dropped downstream. Once it is routed, that is a goal the
    gripper aborts on every tick.
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


# ----------------------------------------------------------------------
# Wire format
# ----------------------------------------------------------------------


def test_frame_round_trips_exactly():
    """Arrays must survive the frame byte for byte — this carries the model's whole input."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (2, 12, 10, 3), dtype=np.uint8)
    agent_pos = rng.uniform(-1, 1, (2, 8))

    channels, header = unpack_observation(
        pack_observation({'camera0_rgb': image, 'agent_pos': agent_pos}, n_obs_steps=2, n_action_steps=16)
    )

    assert np.array_equal(channels['camera0_rgb'], image)
    assert np.array_equal(channels['agent_pos'], agent_pos)
    assert channels['camera0_rgb'].dtype == np.uint8
    assert header['n_obs_steps'] == 2
    assert header['n_action_steps'] == 16


def test_frame_is_smaller_than_the_base64_it_replaced():
    """The whole point of raw bytes: no 4/3 blowup and no intermediate string."""
    image = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    agent_pos = np.zeros((2, 8))
    body = pack_observation({'camera0_rgb': image, 'agent_pos': agent_pos}, n_obs_steps=2, n_action_steps=16)

    payload = image.nbytes + agent_pos.nbytes
    assert payload < len(body) < payload + 1024  # header is a few hundred bytes, not a copy


def test_omitted_channel_is_refused_not_filled_in(client):
    """
    A frame missing a required channel must 422, and say why.

    The format can express omission on purpose — modalities that update slower than the control
    loop are the long-term intent — but nothing caches a channel's last value, so an omitted one
    would reach the model as absent rather than stale. A forward pass absorbs that silently.
    """
    body = pack_observation({'agent_pos': np.zeros((2, 8))}, n_obs_steps=2, n_action_steps=8)
    resp = _post(client, body)

    assert resp.status_code == 422
    detail = resp.json()['detail']
    assert 'camera0_rgb' in detail
    assert 'not yet supported' in detail.lower() or 'NOT yet supported' in detail


def test_truncated_frame_is_refused(client):
    """A body cut short must be a 422 naming the problem, not a reshape into wrong contents."""
    body = _request_body()
    resp = _post(client, body[: len(body) - 16])

    assert resp.status_code == 422
    # The cut lands in whichever blob sits last in the body, so assert on the fault rather than
    # on a channel name that depends on pack order.
    assert 'but the body holds' in resp.json()['detail']


def test_shape_must_agree_with_byte_count():
    """A shape that does not match nbytes is the failure that otherwise decodes into garbage."""
    body = bytearray(pack_observation({'agent_pos': np.zeros((2, 8))}, n_obs_steps=2, n_action_steps=8))
    corrupted = bytes(body).replace(b'"shape": [2, 8]', b'"shape": [2, 7]', 1)
    assert corrupted != bytes(body), 'header text changed shape; update this test'

    with pytest.raises(WireFormatError, match='needs'):
        unpack_observation(corrupted)


def test_obs_wire_copies_have_not_drifted():
    """
    The three copies of obs_wire must be byte identical.

    They live in three separately deployed interpreters with no import path between them — a ROS
    node under /usr/bin/python3, this uv venv, and a conda env inside the training container — so
    duplication is the only option, and silent drift between them is a wire-format mismatch that
    surfaces as a rejected frame at bringup rather than as an error here.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    canonical = root / 'ros2_ws/src/polyumi_ros2/polyumi_ros2/obs_wire.py'
    copies = [
        root / 'inference_server/inference_server/obs_wire.py',
        root / 'external/polyumi_diffusion_policy/obs_wire.py',
    ]
    if not canonical.exists():  # the inference server ships standalone; nothing to compare against
        pytest.skip('running outside the full repo checkout')

    expected = canonical.read_bytes()
    for copy in copies:
        assert copy.exists(), f'{copy} is missing'
        assert copy.read_bytes() == expected, (
            f'{copy.relative_to(root)} has drifted from {canonical.relative_to(root)}. '
            'Copy the canonical file over it rather than editing one side.'
        )

"""
The client and the server, tested against each other.

A real :class:`PolicyClient` puts the exact bytes the ROS node puts on the wire, and the exact
server code a checkpoint runs behind decodes them -- in one process, with no socket and no second
interpreter.

Everything here goes through :func:`create_app`, so it holds for *every* backend. That is the point:
``dummy_server`` is the bringup path, so a frame it accepts must be one a checkpoint would also
accept, and the two share the code that decides.
"""

import numpy as np
import pytest

from polyumi_inference import ActionChunk, Observation, TransportError
from polyumi_inference.backends.sine import SineBackend
from polyumi_inference.server import create_app
from polyumi_inference.testing import in_process_client


class RecordingBackend:
    """A backend that returns a fixed chunk and remembers what it was handed."""

    def __init__(self, n_actions: int = 4):
        """Build a backend whose chunk is n_actions long and distinguishable from zeros."""
        self.seen = []
        self.reset_with = None
        self._chunk = ActionChunk(np.arange(n_actions * 8, dtype=np.float64).reshape(n_actions, 8), model_ms=3.5)

    def predict(self, obs):
        """Record the observation and hand back the fixed chunk."""
        self.seen.append(obs)
        return self._chunk

    def reset(self, agent_pos):
        """Record the episode-start pose."""
        self.reset_with = np.asarray(agent_pos)

    def describe(self):
        """Report ready, in the shape a real backend's /health uses."""
        return {'status': 'ready', 'checkpoint': 'recording'}


def _observation(n_obs_steps: int = 2, n_action_steps: int = 16, **overrides) -> Observation:
    """Build a valid observation; pass a channel by name to replace it, or None to drop it."""
    channels = {
        'camera0_rgb': np.full((n_obs_steps, 6, 6, 3), 200, dtype=np.uint8),
        'agent_pos': np.array([[0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.04]] * n_obs_steps),
    }
    for name, value in overrides.items():
        if value is None:
            channels.pop(name, None)
        else:
            channels[name] = value
    return Observation(channels=channels, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


@pytest.fixture
def backend():
    """Build a backend that records what the app hands it."""
    return RecordingBackend()


@pytest.fixture
def client(backend):
    """Wire a PolicyClient straight into a real app, with the app's lifespan entered."""
    with in_process_client(create_app(backend, title='test')) as policy_client:
        yield policy_client


def test_observation_survives_the_round_trip_into_the_backend(client, backend):
    """
    What the client packs is what the backend is handed -- values, dtypes and window alike.

    This is the assertion the three copied files existed to make true and could not check.
    """
    obs = _observation()
    client.predict(obs)

    (received,) = backend.seen
    assert np.array_equal(received['camera0_rgb'], obs['camera0_rgb'])
    assert received['camera0_rgb'].dtype == np.uint8
    assert np.array_equal(received['agent_pos'], obs['agent_pos'])
    assert (received.n_obs_steps, received.n_action_steps) == (2, 16)


def test_action_chunk_survives_the_round_trip_back(client):
    """And what the backend returns is what the client gets, timings included."""
    chunk = client.predict(_observation())

    assert chunk.actions.shape == (4, 8)
    assert np.array_equal(chunk.actions, np.arange(32, dtype=np.float64).reshape(4, 8))
    assert chunk.model_ms == pytest.approx(3.5)
    # Measured by the app, not the backend, so it is present without the backend doing anything.
    assert chunk.server_total_ms is not None and chunk.server_total_ms >= 0.0


def test_chunk_is_truncated_to_what_was_asked_for(client):
    """The app caps the backend's horizon at n_action_steps so no backend has to remember to."""
    assert client.predict(_observation(n_action_steps=2)).n_action_steps == 2


def test_reset_reaches_the_backend(client, backend):
    """The episode-start pose has to be cached server-side; the frame only carries the current one."""
    client.reset([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.05])

    assert np.allclose(backend.reset_with, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.05])


def test_reset_url_derives_from_the_predict_url(client):
    """One parameter configures all three endpoints, which is how the ROS node is parameterized."""
    assert client.reset_url == 'http://testserver/reset'
    assert client.health_url == 'http://testserver/health'


def test_health_reports_the_backend(client):
    """The app reports reachability; what 'ready' means is the backend's to say."""
    assert client.health() == {'status': 'ready', 'checkpoint': 'recording'}


# ----------------------------------------------------------------------
# Refusals -- asserted from the client's side, which is where they are read
# ----------------------------------------------------------------------


def test_omitted_channel_is_refused_not_filled_in(client):
    """
    A frame missing a required channel must be refused, and say why.

    Omission is expressible on purpose -- modalities that update slower than the control loop are
    the long-term intent -- but nothing caches a channel's last value, so an omitted one would reach
    the model as absent rather than stale. A forward pass absorbs that silently.
    """
    with pytest.raises(TransportError) as excinfo:
        client.predict(_observation(camera0_rgb=None))

    assert excinfo.value.status_code == 422
    assert 'camera0_rgb' in str(excinfo.value)
    assert 'NOT yet supported' in str(excinfo.value)


def test_window_mismatch_is_refused(client):
    """
    The header's window length and an array's leading dim are two claims about the same thing.

    A disagreement means the client packed something other than what it says it packed.
    """
    with pytest.raises(TransportError) as excinfo:
        client.predict(_observation(camera0_rgb=np.zeros((3, 6, 6, 3), dtype=np.uint8)))

    assert excinfo.value.status_code == 422
    assert 'n_obs_steps=2' in str(excinfo.value)


def test_wrong_agent_pos_width_is_refused(client):
    """agent_pos is indexed positionally all the way to the hand, so its width is not negotiable."""
    with pytest.raises(TransportError) as excinfo:
        client.predict(_observation(agent_pos=np.zeros((2, 7))))

    assert excinfo.value.status_code == 422
    assert 'agent_pos must be [To,8]' in str(excinfo.value)


def test_a_non_image_camera_channel_is_refused(client):
    """[To,H,W,3] is what the encoder was trained on; anything else must not reach it."""
    with pytest.raises(TransportError) as excinfo:
        client.predict(_observation(camera0_rgb=np.zeros((2, 6, 6), dtype=np.uint8)))

    assert excinfo.value.status_code == 422
    assert 'camera0_rgb must be [To,H,W,3]' in str(excinfo.value)


def test_a_truncated_body_is_refused(client, backend):
    """A body cut in flight must be refused, not reshaped into plausible wrong contents."""
    app = create_app(backend, title='test')
    with in_process_client(app) as raw_client:
        body = _observation().to_frame()
        reply = raw_client._transport.request('POST', raw_client.predict_url, content=body[:-16], timeout_s=None)

    assert reply.status_code == 422
    assert 'but the body holds' in reply.text
    assert backend.seen == []


def test_the_error_message_carries_the_servers_own_words(client):
    """
    On a 4xx the response body IS the diagnostic; without it this is just '422'.

    The node logs str(the error), so this is the difference between a usable bringup log and a
    status code.
    """
    with pytest.raises(TransportError) as excinfo:
        client.predict(_observation(camera0_rgb=None))

    message = str(excinfo.value)
    assert 'POST http://testserver/predict_cartesian/ failed' in message
    assert 'HTTP 422' in message
    assert excinfo.value.detail and excinfo.value.detail.startswith('Frame omits required channel')


def test_a_dead_server_raises_rather_than_returning_nothing(backend):
    """
    A failed request must not be mistakable for 'no actions this tick'.

    Returning None made those two cases the same value at the call site; they need different
    handling, and one of them needs saying out loud.
    """
    from polyumi_inference.client import PolicyClient

    with PolicyClient('http://127.0.0.1:1/predict_cartesian/', timeout_s=0.5) as offline:
        with pytest.raises(TransportError):
            offline.predict(_observation())


def test_the_dummy_refuses_exactly_what_a_checkpoint_would():
    """
    The bringup server and the real one share the app, so their refusals cannot drift.

    The same frames go to the sine backend and to a stand-in for a checkpoint, and the verdicts
    have to match.
    """
    import os
    from unittest.mock import patch

    bad_frames = [
        _observation(camera0_rgb=None),
        _observation(agent_pos=np.zeros((2, 7))),
        _observation(camera0_rgb=np.zeros((3, 6, 6, 3), dtype=np.uint8)),
    ]

    def verdicts(app):
        out = []
        with in_process_client(app) as c:
            for obs in bad_frames:
                try:
                    c.predict(obs)
                except TransportError as e:
                    out.append((e.status_code, e.detail))
                else:
                    out.append(('ok', None))
        return out

    with patch.dict(os.environ, {'HOME_POSE': '0.56 0.13 0.25 -1 0 0 0 0.05'}):
        dummy = verdicts(create_app(SineBackend.from_env, title='dummy'))
    checkpoint_stand_in = verdicts(create_app(RecordingBackend(), title='real'))

    assert dummy == checkpoint_stand_in
    assert all(status == 422 for status, _ in dummy)

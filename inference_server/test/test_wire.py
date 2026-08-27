"""
Tests for the frame format and the two types both ends of the protocol hold.

These have no server and no client in them: the frame either survives a round trip or it does not,
and a malformed one either names its fault or silently decodes into the wrong numbers.
"""

import json
import struct

import numpy as np
import pytest

from polyumi_inference import ActionChunk, Observation, WireFormatError, pack_frame, unpack_frame


def _observation(n_obs_steps: int = 2, n_action_steps: int = 16) -> Observation:
    """Build a structurally valid observation, small enough to hold in a test."""
    return Observation(
        channels={
            'camera0_rgb': np.full((n_obs_steps, 8, 8, 3), 128, dtype=np.uint8),
            'agent_pos': np.array([[0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.04]] * n_obs_steps),
        },
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )


def _reframe(body: bytes, **header_edits) -> bytes:
    """
    Re-emit a frame with an edited header, keeping the length prefix honest.

    Editing the header bytes in place is not enough: a shorter value leaves the 4-byte prefix
    claiming the old length, so the JSON slice comes out ragged and the frame fails for the wrong
    reason. This rebuilds the prefix, so a corruption test corrupts only what it means to.
    """
    (header_len,) = struct.unpack_from('>I', body)
    header = json.loads(body[4 : 4 + header_len])
    for key, value in header_edits.items():
        if key == 'channels':
            header['channels'].update(value)
        else:
            header[key] = value
    encoded = json.dumps(header).encode('utf-8')
    return struct.pack('>I', len(encoded)) + encoded + body[4 + header_len :]


def test_frame_round_trips_exactly():
    """Arrays must survive the frame byte for byte -- this carries the model's whole input."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (2, 12, 10, 3), dtype=np.uint8)
    agent_pos = rng.uniform(-1, 1, (2, 8))

    obs = Observation.from_frame(
        Observation({'camera0_rgb': image, 'agent_pos': agent_pos}, n_obs_steps=2, n_action_steps=16).to_frame()
    )

    assert np.array_equal(obs['camera0_rgb'], image)
    assert np.array_equal(obs['agent_pos'], agent_pos)
    assert obs['camera0_rgb'].dtype == np.uint8
    assert obs.n_obs_steps == 2
    assert obs.n_action_steps == 16


def test_frame_is_smaller_than_the_base64_it_replaced():
    """The whole point of raw bytes: no 4/3 blowup and no intermediate string."""
    body = _observation(n_obs_steps=2).to_frame()

    obs = Observation.from_frame(body)
    payload = obs['camera0_rgb'].nbytes + obs['agent_pos'].nbytes
    assert payload < len(body) < payload + 1024  # header is a few hundred bytes, not a copy


def test_a_sliced_view_is_serialized_in_the_order_its_shape_claims():
    """
    A non-contiguous array must pack as the reader will read it.

    The obs buffer is built by stacking, but a caller that hands over a transposed or strided view
    would otherwise write bytes in an order the header's shape no longer describes -- and it would
    decode without complaint into plausible, wrong pixels.
    """
    source = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 4)
    view = np.moveaxis(source, 1, -1)  # [2,4,4,3], not contiguous
    assert not view.flags['C_CONTIGUOUS']

    obs = Observation.from_frame(Observation({'camera0_rgb': view}, n_obs_steps=2, n_action_steps=8).to_frame())

    assert np.array_equal(obs['camera0_rgb'], view)


def test_truncated_frame_is_refused():
    """A body cut short must name the problem, not reshape into wrong contents."""
    body = _observation().to_frame()

    with pytest.raises(WireFormatError, match='but the body holds'):
        Observation.from_frame(body[:-16])


def test_shape_must_agree_with_byte_count():
    """A shape that does not match nbytes is the failure that otherwise decodes into garbage."""
    body = Observation({'agent_pos': np.zeros((2, 8))}, n_obs_steps=2, n_action_steps=8).to_frame()
    corrupted = body.replace(b'"shape": [2, 8]', b'"shape": [2, 7]', 1)
    assert corrupted != body, 'header text changed shape; update this test'

    with pytest.raises(WireFormatError, match='needs'):
        Observation.from_frame(corrupted)


def test_garbage_prefix_is_refused_before_allocating():
    """A hostile or corrupt length prefix must be bounded, not believed."""
    with pytest.raises(WireFormatError, match='over the'):
        Observation.from_frame(b'this is not a frame at all, not even close')


@pytest.mark.parametrize('key', ['n_obs_steps', 'n_action_steps'])
def test_window_counts_must_be_positive_ints(key):
    """Everything downstream indexes against these, so a nonsense value is an unreadable frame."""
    corrupted = _reframe(_observation().to_frame(), **{key: 0})

    with pytest.raises(WireFormatError, match=f'{key} must be a positive int'):
        Observation.from_frame(corrupted)


def test_object_dtype_is_refused():
    """Object dtype would have frombuffer reconstruct pointers out of wire bytes."""
    body = pack_frame({'agent_pos': np.zeros((2, 8))}, n_obs_steps=2, n_action_steps=8)
    corrupted = _reframe(body, channels={'agent_pos': {'dtype': '|O', 'shape': [2, 8], 'offset': 0, 'nbytes': 128}})

    with pytest.raises(WireFormatError, match='object dtype'):
        unpack_frame(corrupted)


def test_omitted_channel_is_refused_with_the_reason():
    """
    Omission is expressible on purpose, and refused on purpose.

    Modalities that update slower than the control loop are the long-term intent, but nothing caches
    a channel's last value, so an omitted one would reach the model as absent rather than stale --
    which a forward pass absorbs silently.
    """
    obs = Observation({'agent_pos': np.zeros((2, 8))}, n_obs_steps=2, n_action_steps=8)

    with pytest.raises(WireFormatError) as excinfo:
        obs.require(['camera0_rgb', 'agent_pos'])

    assert 'camera0_rgb' in str(excinfo.value)
    assert 'NOT yet supported' in str(excinfo.value)


# ----------------------------------------------------------------------
# ActionChunk
# ----------------------------------------------------------------------


def test_action_chunk_round_trips_through_json():
    """The response type is one type in both directions; its two halves must agree."""
    chunk = ActionChunk(np.arange(24, dtype=np.float64).reshape(3, 8), model_ms=12.5, server_total_ms=40.0)

    parsed = ActionChunk.from_json(chunk.to_json())

    assert np.array_equal(parsed.actions, chunk.actions)
    assert parsed.model_ms == 12.5
    assert parsed.server_total_ms == 40.0
    assert parsed.n_action_steps == 3


def test_action_chunk_accepts_lists_from_a_backend():
    """Backends build actions from lists; normalizing once means consumers can trust .shape."""
    chunk = ActionChunk([[0.0] * 8, [1.0] * 8])

    assert chunk.actions.shape == (2, 8)
    assert chunk.actions.dtype == np.float64


def test_action_chunk_rejects_a_ragged_reply():
    """Everything downstream indexes positionally -- the gripper is column 7 -- so this fails here."""
    with pytest.raises(WireFormatError, match='actions'):
        ActionChunk.from_json({'actions': [[0.0] * 8, [1.0] * 3]})


def test_absent_timings_stay_absent():
    """A client must be able to tell 'the server did not say' from 'it was zero'."""
    parsed = ActionChunk.from_json({'actions': [[0.0] * 8]})

    assert parsed.model_ms is None
    assert parsed.server_total_ms is None


def test_truncate_keeps_the_timings():
    """Truncation is about how many actions were asked for, not about how long the work took."""
    chunk = ActionChunk(np.zeros((8, 8)), model_ms=7.0, server_total_ms=9.0)

    assert chunk.truncate(3).n_action_steps == 3
    assert chunk.truncate(3).model_ms == 7.0
    assert chunk.truncate(99) is chunk

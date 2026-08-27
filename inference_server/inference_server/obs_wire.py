"""
Binary framing for the ``/predict_cartesian/`` observation request.

The frame is a length-prefixed JSON header followed by the raw bytes of each channel::

    [4B big-endian uint32: header length N][N bytes UTF-8 JSON header][channel blobs]

and the header names every blob's dtype, shape and position::

    {"version": 1,
     "n_obs_steps": 2,
     "n_action_steps": 16,
     "channels": {
       "camera0_rgb": {"dtype": "|u1", "shape": [2,224,224,3], "offset": 0,      "nbytes": 301056},
       "agent_pos":   {"dtype": "<f8", "shape": [2,8],         "offset": 301056, "nbytes": 128}}}

Raw rather than base64 because base64 is 4/3 the bytes and forces an equally large intermediate
string per request. The latency argument is thin — measured on loopback the whole round trip is
2.4 ms with base64 and 1.4 ms without — so the real reason for this format is the shape, not the
speed: an observation is a *set* of channels arriving at different rates, and a flat JSON
document with one hardcoded ``image`` key cannot express that.

**Every channel is a typed blob, including tiny ones.** ``agent_pos`` is 128 bytes and could ride
inside the JSON, but one uniform rule needs no size threshold and no branch on either side.

**Channel names are the dataset's names** (``camera0_rgb``, and later ``finger_rgb`` / ``mic_0``),
so adding a modality is adding a name rather than a name plus a mapping.

Omission is deliberately expressible and deliberately refused; see :func:`require_channels`.

This module is duplicated, byte for byte, into the inference server and the diffusion-policy
fork. They are three separately deployed interpreters — a ROS node under ``/usr/bin/python3``, a
uv venv, and a conda env inside a container — with no import path between them, and this is the
one thing all three must agree on exactly. ``test_dummy_server.py`` fails if the copies drift.
"""

import json
import struct

import numpy as np

#: Frame format version, in the header. Bump only for a change that an old reader would
#: misinterpret rather than reject.
WIRE_VERSION = 1

#: Refuse a header length larger than this before allocating for it. The header is a few hundred
#: bytes in practice; a corrupted or hostile prefix could otherwise claim gigabytes.
MAX_HEADER_BYTES = 1 << 20

_LENGTH_PREFIX = struct.Struct('>I')


class WireFormatError(ValueError):
    """A frame that cannot be decoded, or that omits a channel the policy requires."""


def pack_observation(channels: dict[str, np.ndarray], *, n_obs_steps: int, n_action_steps: int) -> bytes:
    """
    Serialize named arrays into one request frame.

    :param channels: channel name -> array. Order is preserved in the body, which makes a packet
        capture readable, but readers must use the header's offsets rather than assuming it.
    :param n_obs_steps: length of the observation window the arrays carry.
    :param n_action_steps: how many action steps to ask the policy for.
    :return: the frame, ready to POST as ``application/octet-stream``.
    """
    blobs: list[bytes] = []
    meta: dict[str, dict] = {}
    offset = 0
    for name, array in channels.items():
        # ascontiguousarray, not tobytes() alone: a sliced or transposed view would otherwise be
        # serialized in an order the shape no longer describes.
        contiguous = np.ascontiguousarray(array)
        raw = contiguous.tobytes()
        meta[name] = {
            # dtype.str carries byte order explicitly ('<f8', not 'float64'), so the format stays
            # correct if the two ends ever stop sharing a machine.
            'dtype': contiguous.dtype.str,
            'shape': list(contiguous.shape),
            'offset': offset,
            'nbytes': len(raw),
        }
        blobs.append(raw)
        offset += len(raw)

    header = json.dumps(
        {
            'version': WIRE_VERSION,
            'n_obs_steps': int(n_obs_steps),
            'n_action_steps': int(n_action_steps),
            'channels': meta,
        }
    ).encode('utf-8')
    return b''.join([_LENGTH_PREFIX.pack(len(header)), header, *blobs])


def unpack_observation(body: bytes) -> tuple[dict[str, np.ndarray], dict]:
    """
    Decode a request frame into arrays plus the header that described them.

    Every failure is a :class:`WireFormatError` naming what was wrong, because the callers turn
    these into 422s: a malformed frame is a bad request, not a server fault.

    :param body: the complete request body.
    :return: ``(channels, header)``. The arrays are views onto ``body`` and are read-only; copy
        before mutating. Not copying is the point — it is what makes the raw framing cheaper
        than base64 rather than merely narrower.
    """
    if len(body) < _LENGTH_PREFIX.size:
        raise WireFormatError(f'Frame is {len(body)} bytes, too short to hold a header length')
    (header_len,) = _LENGTH_PREFIX.unpack_from(body)
    if header_len > MAX_HEADER_BYTES:
        raise WireFormatError(f'Header claims {header_len} bytes, over the {MAX_HEADER_BYTES} limit')
    blob_start = _LENGTH_PREFIX.size + header_len
    if len(body) < blob_start:
        raise WireFormatError(f'Header claims {header_len} bytes but only {len(body) - _LENGTH_PREFIX.size} follow')

    try:
        header = json.loads(body[_LENGTH_PREFIX.size : blob_start])
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise WireFormatError(f'Header is not valid JSON: {e}') from e
    if not isinstance(header, dict):
        raise WireFormatError(f'Header must be a JSON object, got {type(header).__name__}')

    version = header.get('version')
    if version != WIRE_VERSION:
        raise WireFormatError(f'Unsupported frame version {version!r}; this build speaks {WIRE_VERSION}')

    raw_channels = header.get('channels')
    if not isinstance(raw_channels, dict):
        raise WireFormatError("Header has no 'channels' object")

    blobs = memoryview(body)[blob_start:]
    channels: dict[str, np.ndarray] = {}
    for name, spec in raw_channels.items():
        channels[name] = _channel_array(name, spec, blobs)
    return channels, header


def _channel_array(name: str, spec, blobs: memoryview) -> np.ndarray:
    """Validate one channel's header entry and return its array view."""
    if not isinstance(spec, dict):
        raise WireFormatError(f'Channel {name!r} descriptor must be an object, got {type(spec).__name__}')
    missing = {'dtype', 'shape', 'offset', 'nbytes'} - spec.keys()
    if missing:
        raise WireFormatError(f'Channel {name!r} descriptor is missing {sorted(missing)}')

    try:
        dtype = np.dtype(spec['dtype'])
    except TypeError as e:
        raise WireFormatError(f'Channel {name!r} has unusable dtype {spec["dtype"]!r}: {e}') from e
    # Object dtype would make frombuffer reconstruct pointers out of wire bytes.
    if dtype.hasobject:
        raise WireFormatError(f'Channel {name!r} declares object dtype {spec["dtype"]!r}, which is refused')

    shape = spec['shape']
    if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
        raise WireFormatError(f'Channel {name!r} has a non-integer or negative shape: {shape!r}')

    offset, nbytes = spec['offset'], spec['nbytes']
    if not isinstance(offset, int) or not isinstance(nbytes, int) or offset < 0 or nbytes < 0:
        raise WireFormatError(f'Channel {name!r} has a non-integer or negative offset/nbytes')
    if offset + nbytes > len(blobs):
        raise WireFormatError(f'Channel {name!r} runs to byte {offset + nbytes} but the body holds {len(blobs)}')

    # The shape is what the reader will trust, so it has to agree with the byte count. Without
    # this a truncated blob reshapes into a plausible array of the wrong contents.
    expected = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
    if nbytes != expected:
        raise WireFormatError(
            f'Channel {name!r} declares {nbytes} bytes but shape {shape} of {dtype.str} needs {expected}'
        )

    return np.frombuffer(blobs, dtype=dtype, count=int(np.prod(shape)) if shape else 1, offset=offset).reshape(shape)


def require_channels(channels: dict[str, np.ndarray], required) -> None:
    """
    Refuse a frame that omits a channel the policy needs.

    The frame format can express any subset of channels, and that is deliberate: modalities
    arrive at different rates (the finger camera at 10 fps, the contact mic continuously) and the
    long-term intent is for a request to carry only what is new. **That is not implemented.** The
    policy consumes a full observation window every step, and nothing here caches a channel's last
    value to fill the gap, so an omitted channel would reach the model as absent rather than as
    stale — which a forward pass absorbs silently instead of rejecting.

    So the shape is in the format and the boundary is in this error, rather than the boundary
    being a surprise at the far end of a rollout.

    :param channels: what the frame carried.
    :param required: channel names the policy cannot run without.
    :raises WireFormatError: naming every missing channel.
    """
    missing = [name for name in required if name not in channels]
    if missing:
        raise WireFormatError(
            f'Frame omits required channel(s) {missing}; it carried {sorted(channels)}. '
            'Per-channel omission is part of this wire format by design, for modalities that '
            'update slower than the control loop, but it is NOT yet supported server-side: the '
            'policy needs a full observation window every step and no last-value cache exists. '
            'Send every required channel on every request.'
        )

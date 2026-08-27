"""
The PolyUMI inference protocol, and both ends of it.

One library owns the wire format, the client that speaks it, and the server app that answers --
because the three are one contract and used to be three copies of a file. The ROS-side
``policy_client_node`` imports the client; the dummy server and the diffusion-policy fork's
``serve_policy.py`` are backends behind :func:`~polyumi_inference.server.create_app`.

Layers, innermost first:

- :mod:`polyumi_inference.wire` -- how bytes become arrays.
- :mod:`polyumi_inference.types` -- :class:`Observation` and :class:`ActionChunk`, the two things
  both ends hold.
- :mod:`polyumi_inference.contract` -- which arrays the PolyUMI policy actually requires.
- :mod:`polyumi_inference.client` / :mod:`polyumi_inference.server` -- the two ends.

Only ``client`` needs ``requests`` and only ``server`` needs fastapi/pydantic, and each imports its
own; installing this library for one end does not drag in the other's dependencies. The top-level
names below are therefore the dependency-free core -- import the client and server modules directly.

**Python 3.9 compatible on purpose.** The diffusion-policy container's conda env is 3.9 with numpy
1.24, while the ROS node is 3.12; this library has to import under both, so every module carries
``from __future__ import annotations`` and none uses 3.10+ syntax.
"""

from __future__ import annotations

from polyumi_inference.contract import (
    AGENT_POS_CHANNEL,
    AGENT_POS_DIM,
    CAMERA_CHANNEL,
    REQUIRED_CHANNELS,
    validate_observation,
)
from polyumi_inference.errors import PolyumiInferenceError, TransportError, WireFormatError
from polyumi_inference.types import ActionChunk, Observation
from polyumi_inference.wire import WIRE_VERSION, pack_frame, unpack_frame

__all__ = [
    'AGENT_POS_CHANNEL',
    'AGENT_POS_DIM',
    'CAMERA_CHANNEL',
    'REQUIRED_CHANNELS',
    'WIRE_VERSION',
    'ActionChunk',
    'Observation',
    'PolyumiInferenceError',
    'TransportError',
    'WireFormatError',
    'pack_frame',
    'unpack_frame',
    'validate_observation',
]

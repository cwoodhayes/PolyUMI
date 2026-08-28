"""
The two types the client and the server both hold.

An :class:`Observation` goes one way and an :class:`ActionChunk` comes back, and each is built by
one end and consumed by the other. They are defined once, here, so the two ends cannot disagree
about them -- which is the whole reason this library exists.

Plain dataclasses rather than pydantic models, deliberately. These are the types the ROS node
imports, and apt on Ubuntu 24.04 ships ``python3-pydantic`` 1.10, while fastapi needs v2: a
pydantic field here would put a v1/v2 split under ``/usr/bin/python3``. They also hold numpy
arrays, which pydantic can only wave through with ``arbitrary_types_allowed``, and the validation
that actually matters -- dtype and shape agreeing with the byte count -- is not expressible as a
schema anyway (it lives in :mod:`polyumi_inference.wire`). pydantic still earns its keep on the
server, where it parses untrusted JSON and shapes the OpenAPI response; see
:mod:`polyumi_inference.server`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from polyumi_inference.errors import WireFormatError
from polyumi_inference.wire import header_count, pack_frame, unpack_frame

# eq=False on both: the dataclass __eq__ would compare numpy arrays with ==, which yields an array
# and then raises on bool(). Identity comparison is what callers actually want here; tests compare
# the arrays themselves with np.array_equal.


@dataclass(frozen=True, eq=False)
class Observation:
    """
    One observation window, as a set of named channels plus the window's own dimensions.

    :param channels: channel name -> array, each ``[n_obs_steps, ...]``. Names are the dataset's
        (``camera0_rgb``, ``agent_pos``), so adding a modality is adding an entry.
    :param n_obs_steps: length of the observation window.
    :param n_action_steps: how many action steps the requester wants back.
    """

    channels: Mapping[str, np.ndarray]
    n_obs_steps: int
    n_action_steps: int

    def to_frame(self) -> bytes:
        """Serialize to the binary request body (see :mod:`polyumi_inference.wire`)."""
        return pack_frame(
            self.channels,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
        )

    @classmethod
    def from_frame(cls, body: bytes) -> Observation:
        """
        Decode a request body. Raises :class:`WireFormatError` on anything unreadable.

        The arrays are read-only views onto ``body``; nothing is copied.
        """
        channels, header = unpack_frame(body)
        return cls(
            channels=channels,
            n_obs_steps=header_count(header, 'n_obs_steps'),
            n_action_steps=header_count(header, 'n_action_steps'),
        )

    def require(self, names: Iterable[str]) -> None:
        """
        Refuse an observation that omits a channel the policy needs.

        The frame format can express any subset of channels, and that is deliberate: modalities
        arrive at different rates (the finger camera at 10 fps, the contact mic continuously) and
        the long-term intent is for a request to carry only what is new. **That is not
        implemented.** The policy consumes a full observation window every step, and nothing here
        caches a channel's last value to fill the gap, so an omitted channel would reach the model
        as absent rather than as stale -- which a forward pass absorbs silently instead of
        rejecting.

        So the shape is in the format and the boundary is in this error, rather than the boundary
        being a surprise at the far end of a rollout.

        :param names: channel names the policy cannot run without.
        :raises WireFormatError: naming every missing channel.
        """
        missing = [name for name in names if name not in self.channels]
        if missing:
            raise WireFormatError(
                f'Frame omits required channel(s) {missing}; it carried {sorted(self.channels)}. '
                'Per-channel omission is part of this wire format by design, for modalities that '
                'update slower than the control loop, but it is NOT yet supported server-side: the '
                'policy needs a full observation window every step and no last-value cache exists. '
                'Send every required channel on every request.'
            )

    def __getitem__(self, name: str) -> np.ndarray:
        return self.channels[name]

    def __contains__(self, name: object) -> bool:
        return name in self.channels

    def names(self) -> list:
        """Channel names carried, sorted -- for error messages and assertions."""
        return sorted(self.channels)


#: Width of a wire pose, ``[x, y, z, qx, qy, qz, qw, gripper]`` -- an :class:`ActionChunk` row and
#: an :class:`Observation`'s ``agent_pos`` channel are both this wide. Defined here, not in
#: :mod:`polyumi_inference.contract`, because :class:`ActionChunk` needs it and importing it from
#: ``contract`` would be circular (``contract`` already imports :class:`Observation` from this
#: module); ``contract`` imports it back from here instead of holding its own copy.
AGENT_POS_DIM = 8


@dataclass(frozen=True, eq=False)
class ActionChunk:
    """
    A chunk of absolute end-effector actions, plus how long producing it took.

    One type in both directions: a backend returns it, :func:`polyumi_inference.server.create_app`
    stamps ``server_total_ms`` and serializes it, and :class:`polyumi_inference.client.PolicyClient`
    parses it back. Two hand-matched schemas is exactly the drift this library exists to remove.

    :param actions: ``[Ta, 8]`` -- ``[x, y, z, qx, qy, qz, qw, gripper]`` in the robot base frame.
    :param model_ms: the forward pass alone. Filled by the backend, which is the only place it can
        be measured honestly (through the CUDA sync point). ``None`` means the server did not say,
        which a client must not confuse with zero.
    :param server_total_ms: wall time the server process spent on the request. Filled by the app.
    """

    actions: np.ndarray = field()
    model_ms: float | None = None
    server_total_ms: float | None = None

    def __post_init__(self) -> None:
        # Backends build these from lists as readily as from arrays; normalize once here so every
        # consumer can rely on .shape rather than guessing.
        actions = np.asarray(self.actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != AGENT_POS_DIM:
            raise ValueError(f'actions must be [Ta,{AGENT_POS_DIM}], got shape {list(actions.shape)}')
        # This is the last stop before the ROS side indexes columns 0-7 straight into pose/gripper
        # commands -- a NaN or Inf here (a starved forward pass, a bad checkpoint) must not reach
        # the robot as a number that merely looks like a pose.
        if not np.isfinite(actions).all():
            raise ValueError('actions contains a NaN or Inf value')
        object.__setattr__(self, 'actions', actions)

    @property
    def n_action_steps(self) -> int:
        """Number of actions carried."""
        return int(self.actions.shape[0])

    def truncate(self, n: int) -> ActionChunk:
        """Return the first ``n`` actions, keeping the timings. A no-op if the chunk is shorter."""
        if n >= self.n_action_steps:
            return self
        return ActionChunk(self.actions[:n], model_ms=self.model_ms, server_total_ms=self.server_total_ms)

    def to_json(self) -> dict:
        """Render the response body, as the wire carries it."""
        return {
            'actions': self.actions.tolist(),
            'n_action_steps': self.n_action_steps,
            'server_total_ms': self.server_total_ms,
            'model_ms': self.model_ms,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> ActionChunk:
        """
        Parse a response body. Raises :class:`WireFormatError` if it is not one.

        Kept strict on ``actions`` because everything downstream indexes it positionally -- the
        gripper is column 7 -- so a ragged or 1-D reply must fail here rather than at the hand.
        """
        actions: Sequence = doc.get('actions')  # type: ignore[assignment]
        if actions is None:
            raise WireFormatError(f"Response has no 'actions'; it carried {sorted(doc)}")
        try:
            chunk = cls(
                np.asarray(actions, dtype=np.float64).reshape(len(actions), -1)
                if len(actions)
                else np.empty((0, AGENT_POS_DIM)),
                model_ms=_opt_float(doc.get('model_ms')),
                server_total_ms=_opt_float(doc.get('server_total_ms')),
            )
        except (ValueError, TypeError) as e:
            raise WireFormatError(f'Response actions are not a [Ta, action_dim] array: {e}') from e
        return chunk


def _opt_float(value: Any) -> float | None:
    """Coerce a nullable timing field; a non-numeric value is treated as absent, not as zero."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

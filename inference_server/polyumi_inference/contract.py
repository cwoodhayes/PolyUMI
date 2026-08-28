"""
What the PolyUMI policy requires of an observation.

:mod:`polyumi_inference.wire` decides whether a frame is *readable*. This module decides whether it
is *servable* -- the channels, dimensions and shapes a checkpoint was trained on.

It exists as one function because two servers must agree on it exactly. ``dummy_server`` is the
bringup path: every frame it accepts is one a checkpoint will later be handed, so a frame it accepts
and the real server rejects is a bug discovered on hardware. Both call
:func:`validate_observation`, so that agreement is a shared call rather than two matching listings.
"""

from __future__ import annotations

from typing import Iterable

from polyumi_inference.errors import WireFormatError
from polyumi_inference.types import AGENT_POS_DIM, Observation

#: The wrist camera, named for the dataset field it comes from.
CAMERA_CHANNEL = 'camera0_rgb'
#: Absolute end-effector pose ``[x, y, z, qx, qy, qz, qw, gripper]``. A wire concept: the server
#: splits it into three separate ``shape_meta`` fields before the policy sees it.
AGENT_POS_CHANNEL = 'agent_pos'

#: Channels the policy cannot run without. Named for the dataset's own fields, so wiring a new
#: modality is adding a name here and in shape_meta rather than reshaping the request.
REQUIRED_CHANNELS = (CAMERA_CHANNEL, AGENT_POS_CHANNEL)


def validate_observation(obs: Observation, required: Iterable[str] = REQUIRED_CHANNELS) -> None:
    """
    Check an observation against the policy's contract, or raise.

    :raises WireFormatError: with a message naming the specific disagreement. The caller turns
        these into 422s: an observation the policy cannot consume is a bad request.
    """
    obs.require(required)

    # The header's window length and each array's leading dim are two independent claims about the
    # same thing; a mismatch means the client packed something other than what it says it packed.
    # ndim first: a 0-d channel (shape ()) has no [0] to read, and shape[0] on it raises IndexError
    # rather than the WireFormatError this is supposed to turn a bad frame into.
    for name in required:
        arr = obs[name]
        if arr.ndim == 0 or arr.shape[0] != obs.n_obs_steps:
            raise WireFormatError(f'{name} leading dim must be n_obs_steps={obs.n_obs_steps}, got {list(arr.shape)}')

    if CAMERA_CHANNEL in obs:
        image = obs[CAMERA_CHANNEL]
        if image.ndim != 4 or image.shape[-1] != 3:
            raise WireFormatError(f'{CAMERA_CHANNEL} must be [To,H,W,3], got {list(image.shape)}')

    if AGENT_POS_CHANNEL in obs:
        agent_pos = obs[AGENT_POS_CHANNEL]
        if agent_pos.ndim != 2 or agent_pos.shape[1] != AGENT_POS_DIM:
            raise WireFormatError(f'{AGENT_POS_CHANNEL} must be [To,{AGENT_POS_DIM}], got {list(agent_pos.shape)}')

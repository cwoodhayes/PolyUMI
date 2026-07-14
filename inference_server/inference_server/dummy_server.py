"""
Dummy inference server for PolyUMI policy client bringup.

Implements the /predict_cartesian/ endpoint with a sine-wave oscillator instead of a real
policy, so the ROS2 policy_client_node can be developed and tested end-to-end without a
trained checkpoint.

Usage:
    HOME_POSE="0.4 0.0 0.4 0 0 0 1 0.04" uv run uvicorn inference_server.dummy_server:app --host 0.0.0.0 --port 8000
"""

import base64
import math
import os
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

REQUIRED_OBS_KEYS = {'image', 'agent_pos'}
AGENT_POS_DIM = 8  # [x, y, z, qx, qy, qz, qw, gripper_width]
OSCILLATION_AMPLITUDE_M = 0.05
OSCILLATION_PERIOD_STEPS = 20  # full cycle over this many /predict calls
DEFAULT_HOME_POSE = '0.56 0.13 0.25 -1 0 0 0 0.4'  # xyz qxqyqzqw gripper


class PredictRequest(BaseModel):
    """Request body for /predict_cartesian/."""

    n_obs_steps: Annotated[int, Field(ge=1)] = 2
    n_action_steps: Annotated[int, Field(ge=1)] = 1
    observations: dict


class PredictResponse(BaseModel):
    """Response body for /predict_cartesian/."""

    actions: list[list[float]]
    n_action_steps: int


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_call_count: int = 0
_home_pose: np.ndarray = np.empty(shape=(1,))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _home_pose
    raw = os.environ.get('HOME_POSE', DEFAULT_HOME_POSE)
    vals = [float(v) for v in raw.split()]
    if len(vals) != AGENT_POS_DIM:
        raise ValueError(f'HOME_POSE must have {AGENT_POS_DIM} values (xyz qxqyqzqw gripper), got {len(vals)}')
    _home_pose = np.array(vals)
    yield


app = FastAPI(title='PolyUMI Dummy Inference Server', lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post('/predict_cartesian/', response_model=PredictResponse)
def predict_cartesian(req: PredictRequest) -> PredictResponse:
    """Return an n_action_steps-long chunk of a sinusoidally oscillating EEF pose."""
    global _call_count

    # Validate observation keys
    missing = REQUIRED_OBS_KEYS - req.observations.keys()
    if missing:
        raise HTTPException(status_code=422, detail=f'Missing observation keys: {missing}')

    # Validate image: base64-encoded raw bytes + dtype/shape (see policy_client_node._control_tick)
    image = req.observations.get('image')
    if not isinstance(image, dict) or not {'dtype', 'shape', 'data'} <= image.keys():
        raise HTTPException(status_code=422, detail="image must be a dict with 'dtype', 'shape', 'data'")
    try:
        image_bytes = base64.b64decode(image['data'])
        image_arr = np.frombuffer(image_bytes, dtype=np.dtype(image['dtype'])).reshape(image['shape'])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'Failed to decode image: {e}') from e
    if image_arr.shape[0] != req.n_obs_steps:
        raise HTTPException(
            status_code=422,
            detail=f'image leading dim must be n_obs_steps={req.n_obs_steps}, got {image_arr.shape[0]}',
        )

    # Validate agent_pos shape: [n_obs_steps, AGENT_POS_DIM]
    agent_pos = req.observations.get('agent_pos')
    if (
        not isinstance(agent_pos, list)
        or len(agent_pos) != req.n_obs_steps
        or not all(isinstance(row, list) and len(row) == AGENT_POS_DIM for row in agent_pos)
    ):
        raise HTTPException(
            status_code=422,
            detail=f'agent_pos must have shape [{req.n_obs_steps}, {AGENT_POS_DIM}]',
        )

    # Oscillate X around the fixed home pose (set via HOME_POSE env var at startup).
    # Return a genuine forward-looking chunk — one pose per step, phase advancing by one
    # OSCILLATION_PERIOD_STEPS-th per step — rather than n_return copies of a single pose,
    # so the client has an actual multi-waypoint path to plan+execute (not n identical
    # points). _call_count advances by the full chunk length so consecutive calls continue
    # the sine smoothly instead of restarting from the same phase.
    model_n_action_steps = 8  # matches training config n_action_steps
    n_return = min(req.n_action_steps, model_n_action_steps)

    actions = []
    for i in range(n_return):
        phase = 2 * math.pi * (_call_count + i) / OSCILLATION_PERIOD_STEPS
        delta_x = OSCILLATION_AMPLITUDE_M * math.sin(phase)
        target = _home_pose.copy()
        target[0] += delta_x
        actions.append(target.tolist())
    _call_count += n_return

    return PredictResponse(actions=actions, n_action_steps=n_return)


# ---------------------------------------------------------------------------
# Entry point (for `uv run dummy-server`)
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the dummy server via uvicorn."""
    import uvicorn

    uvicorn.run('inference_server.dummy_server:app', host='0.0.0.0', port=8000, reload=False)

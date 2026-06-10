"""
Dummy inference server for PolyUMI policy client bringup.

Implements the /predict_cartesian/ endpoint with a sine-wave oscillator instead of a real
policy, so the ROS2 policy_client_node can be developed and tested end-to-end without a
trained checkpoint.

Usage:
    HOME_POSE="0.4 0.0 0.4 0 0 0 1 0.04" uv run uvicorn inference_server.dummy_server:app --host 0.0.0.0 --port 8000
"""

import math
import os
from typing import Annotated

import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

REQUIRED_OBS_KEYS = {'image', 'agent_pos'}
AGENT_POS_DIM = 8  # [x, y, z, qx, qy, qz, qw, gripper_width]
OSCILLATION_AMPLITUDE_M = 0.05
OSCILLATION_PERIOD_STEPS = 20  # full cycle over this many /predict calls


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
_home_pose: np.ndarray = np.array([0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0, 0.04])


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _home_pose
    raw = os.environ.get('HOME_POSE', '0.4 0.0 0.4 0 0 0 1 0.04')
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
    """Return a sinusoidally oscillating EEF pose for bringup testing."""
    global _call_count

    # Validate observation keys
    missing = REQUIRED_OBS_KEYS - req.observations.keys()
    if missing:
        raise HTTPException(status_code=422, detail=f'Missing observation keys: {missing}')

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

    # Oscillate X around the most recent agent_pos (ignoring home pose env var at runtime)
    current_pose = np.array(agent_pos[-1])
    phase = 2 * math.pi * _call_count / OSCILLATION_PERIOD_STEPS
    delta_x = OSCILLATION_AMPLITUDE_M * math.sin(phase)
    _call_count += 1

    target = current_pose.copy()
    target[0] += delta_x

    model_n_action_steps = 8  # matches training config n_action_steps
    n_return = min(req.n_action_steps, model_n_action_steps)
    actions = [target.tolist() for _ in range(n_return)]

    return PredictResponse(actions=actions, n_action_steps=n_return)


# ---------------------------------------------------------------------------
# Entry point (for `uv run dummy-server`)
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the dummy server via uvicorn."""
    import uvicorn

    uvicorn.run('inference_server.dummy_server:app', host='0.0.0.0', port=8000, reload=False)

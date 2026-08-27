"""
Dummy inference server for PolyUMI policy client bringup.

Implements the /predict_cartesian/ endpoint with a sine-wave oscillator instead of a real
policy, so the ROS2 policy_client_node can be developed and tested end-to-end without a
trained checkpoint.

Two channels oscillate, at the same frequency but 90 degrees apart: X on a sine, gripper width
on a cosine. The phase offset is deliberate — the gripper's extremes land on X's zero crossings,
so a routing bug that feeds X into the gripper (or vice versa) is visible at a glance in the logs
and in Foxglove, instead of looking perfectly plausible.

Usage:
    HOME_POSE="0.4 0.0 0.4 0 0 0 1 0.04" uv run uvicorn inference_server.dummy_server:app --host 0.0.0.0 --port 8000
"""

import base64
import math
import os
import time
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
# Gripper swing, in the same units as the training data: metres of opening from fully closed, so
# 0 is shut. NOT the robot's jaw aperture — policy_client_node adds the FR3's closed aperture back
# on and clamps (see polyumi_ros2/gripper_map.py). Centred on the home width, so keep the amplitude
# <= that to stay non-negative.
GRIPPER_OSCILLATION_AMPLITUDE_M = 0.04
DEFAULT_HOME_POSE = '0.56 0.13 0.25 -1 0 0 0 0.05'  # xyz qxqyqzqw gripper
# Sanity bound on the home gripper width. The Franka Hand tops out near 0.0817 m and the handheld
# gripper's tags separate to ~0.1 m, so anything past this is a units error (this default used to
# read 0.4 — 400 mm — which went unnoticed only because the width was being dropped downstream).
MAX_PLAUSIBLE_GRIPPER_M = 0.2


class PredictRequest(BaseModel):
    """Request body for /predict_cartesian/."""

    n_obs_steps: Annotated[int, Field(ge=1)] = 2
    n_action_steps: Annotated[int, Field(ge=1)] = 1
    observations: dict


class PredictResponse(BaseModel):
    """Response body for /predict_cartesian/."""

    actions: list[list[float]]
    n_action_steps: int
    #: Wall time this process spent on the request, in ms. Carried so the dummy speaks the same
    #: contract as serve_policy.py — policy_client_node plots the round trip split into server
    #: and network, and a dummy that omitted the field would leave that plot blank during bringup.
    server_total_ms: float | None = None
    #: The "forward pass", in ms. Near zero here; the field exists for contract parity.
    model_ms: float | None = None


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
    gripper = vals[-1]
    if not 0.0 < gripper <= MAX_PLAUSIBLE_GRIPPER_M:
        raise ValueError(
            f'HOME_POSE gripper width {gripper} m is out of range (0, {MAX_PLAUSIBLE_GRIPPER_M}]. '
            'The last HOME_POSE value is a width in METRES — a value like 0.4 is 400 mm, ~5x the '
            "Franka Hand's stroke. Did you mean 0.04?"
        )
    _home_pose = np.array(vals)
    yield


app = FastAPI(title='PolyUMI Dummy Inference Server', lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post('/reset')
def reset(body: dict | None = None) -> dict:
    """
    No-op reset for contract parity with the real serve_policy server.

    The real server caches the episode-start pose here; the dummy oscillator is stateless and
    ignores the pose, but the endpoint must exist so policy_client_node's per-episode /reset call
    doesn't 404 against the dummy.
    """
    return {'status': 'ok'}


@app.post('/predict_cartesian/', response_model=PredictResponse)
def predict_cartesian(req: PredictRequest) -> PredictResponse:
    """Return an n_action_steps-long chunk of a sinusoidally oscillating EEF pose."""
    global _call_count

    t_start = time.perf_counter()

    # Validate observation keys
    missing = REQUIRED_OBS_KEYS - req.observations.keys()
    if missing:
        raise HTTPException(status_code=422, detail=f'Missing observation keys: {missing}')

    # Validate image: base64-encoded raw bytes + dtype/shape, uint8 [To,H,W,3] in practice
    # (see policy_client_node._control_tick). The dtype is honoured rather than assumed, so the
    # dummy keeps accepting whatever the real server does.
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

    # Oscillate X and the gripper width around the fixed home pose (set via HOME_POSE env var at
    # startup). Return a genuine forward-looking chunk — one pose per step, phase advancing by one
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
        # cos against X's sin — same frequency, quarter period apart. See the module docstring:
        # this is what makes a gripper/pose routing mix-up visible rather than plausible.
        delta_grip = GRIPPER_OSCILLATION_AMPLITUDE_M * math.cos(phase)
        target = _home_pose.copy()
        target[0] += delta_x
        # Clamp at 0: a negative width is meaningless, and the client would clamp it anyway.
        target[7] = max(0.0, target[7] + delta_grip)
        actions.append(target.tolist())
    _call_count += n_return

    return PredictResponse(
        actions=actions,
        n_action_steps=n_return,
        server_total_ms=(time.perf_counter() - t_start) * 1e3,
        model_ms=0.0,
    )


# ---------------------------------------------------------------------------
# Entry point (for `uv run dummy-server`)
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the dummy server via uvicorn."""
    import uvicorn

    uvicorn.run('inference_server.dummy_server:app', host='0.0.0.0', port=8000, reload=False)

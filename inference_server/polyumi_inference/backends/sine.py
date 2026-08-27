"""
A sine-wave backend, so the ROS client can be brought up without a GPU or a checkpoint.

Two channels oscillate, at the same frequency but 90 degrees apart: X on a sine, gripper width on a
cosine. The phase offset is deliberate -- the gripper's extremes land on X's zero crossings, so a
routing bug that feeds X into the gripper (or vice versa) is visible at a glance in the logs and in
Foxglove, instead of looking perfectly plausible.

Because it goes through :func:`polyumi_inference.server.create_app` like the real server, it refuses
exactly what the real server refuses. That matters: this is the bringup path, so a frame it accepts
must be one a checkpoint would also accept.
"""

from __future__ import annotations

import math
import os

import numpy as np

from polyumi_inference.contract import AGENT_POS_DIM
from polyumi_inference.types import ActionChunk, Observation

OSCILLATION_AMPLITUDE_M = 0.05
OSCILLATION_PERIOD_STEPS = 20  # full cycle over this many /predict calls
# Gripper swing, in the same units as the training data: metres of opening from fully closed, so
# 0 is shut. NOT the robot's jaw aperture -- policy_client_node adds the FR3's closed aperture back
# on and clamps (see polyumi_ros2/gripper_map.py). Centred on the home width, so keep the amplitude
# <= that to stay non-negative.
GRIPPER_OSCILLATION_AMPLITUDE_M = 0.04
DEFAULT_HOME_POSE = '0.56 0.13 0.25 -1 0 0 0 0.05'  # xyz qxqyqzqw gripper
# Sanity bound on the home gripper width. The Franka Hand tops out near 0.0817 m and the handheld
# gripper's tags separate to ~0.1 m, so anything past this is a units error.
MAX_PLAUSIBLE_GRIPPER_M = 0.2
#: The horizon a trained checkpoint emits. Matched here so the dummy's chunks are the same length
#: the client will see in production, which is what the stale-action arithmetic is tuned against.
MODEL_N_ACTION_STEPS = 8


class SineBackend:
    """Oscillates the end-effector's X and the gripper width about a fixed home pose."""

    def __init__(self, home_pose: np.ndarray) -> None:
        """Build the oscillator centred on ``home_pose`` (an 8-vector wire pose)."""
        self._home_pose = np.asarray(home_pose, dtype=np.float64)
        self._call_count = 0

    @classmethod
    def from_env(cls) -> SineBackend:
        """
        Build from ``$HOME_POSE``, refusing an implausible one.

        Raising here means a misconfigured server fails at startup rather than commanding a goal
        the hand aborts on every tick.
        """
        raw = os.environ.get('HOME_POSE', DEFAULT_HOME_POSE)
        vals = [float(v) for v in raw.split()]
        if len(vals) != AGENT_POS_DIM:
            raise ValueError(f'HOME_POSE must have {AGENT_POS_DIM} values (xyz qxqyqzqw gripper), got {len(vals)}')
        gripper = vals[-1]
        if not 0.0 < gripper <= MAX_PLAUSIBLE_GRIPPER_M:
            raise ValueError(
                f'HOME_POSE gripper width {gripper} m is out of range (0, {MAX_PLAUSIBLE_GRIPPER_M}]. '
                'The last HOME_POSE value is a width in METRES -- a value like 0.4 is 400 mm, ~5x the '
                "Franka Hand's stroke. Did you mean 0.04?"
            )
        return cls(np.array(vals))

    def reset(self, agent_pos: np.ndarray) -> None:
        """No-op: the oscillator is stateless with respect to where the episode started."""

    def describe(self) -> dict:
        """Report readiness in the same shape the real server's ``/health`` uses."""
        return {'status': 'ready', 'checkpoint': None, 'device': 'cpu', 'episode_start_set': False}

    def predict(self, obs: Observation) -> ActionChunk:
        """
        Return a forward-looking chunk of an oscillating EEF pose.

        One pose per step with the phase advancing, rather than N copies of a single pose, so the
        client has a real multi-waypoint path to plan and execute. ``_call_count`` advances by the
        full chunk length so consecutive calls continue the sine smoothly instead of restarting from
        the same phase.
        """
        n_return = min(obs.n_action_steps, MODEL_N_ACTION_STEPS)
        actions = []
        for i in range(n_return):
            phase = 2 * math.pi * (self._call_count + i) / OSCILLATION_PERIOD_STEPS
            delta_x = OSCILLATION_AMPLITUDE_M * math.sin(phase)
            # cos against X's sin -- same frequency, quarter period apart. See the module docstring:
            # this is what makes a gripper/pose routing mix-up visible rather than plausible.
            delta_grip = GRIPPER_OSCILLATION_AMPLITUDE_M * math.cos(phase)
            target = self._home_pose.copy()
            target[0] += delta_x
            # Clamp at 0: a negative width is meaningless, and the client would clamp it anyway.
            target[7] = max(0.0, target[7] + delta_grip)
            actions.append(target)
        self._call_count += n_return

        return ActionChunk(np.asarray(actions, dtype=np.float64).reshape(n_return, AGENT_POS_DIM), model_ms=0.0)

"""
Dummy inference server for PolyUMI policy client bringup.

The ROS2 ``policy_client_node`` can be developed and tested end to end against this instead of a
trained checkpoint. It is the same app as the real server -- same routes, same frame decoding, same
refusals -- with :class:`polyumi_inference.backends.sine.SineBackend` in place of a policy.

Usage:
    HOME_POSE="0.4 0.0 0.4 0 0 0 1 0.04" uv run dummy-server
"""

from __future__ import annotations

from polyumi_inference.backends.sine import (
    DEFAULT_HOME_POSE,
    GRIPPER_OSCILLATION_AMPLITUDE_M,
    MAX_PLAUSIBLE_GRIPPER_M,
    OSCILLATION_AMPLITUDE_M,
    OSCILLATION_PERIOD_STEPS,
    SineBackend,
)
from polyumi_inference.server import create_app, serve

# from_env, not an instance: create_app calls it at startup, so a nonsense HOME_POSE fails when the
# server starts rather than being commanded at the hand.
app = create_app(SineBackend.from_env, title='PolyUMI Dummy Inference Server')


def main() -> None:
    """Launch the dummy server via uvicorn."""
    serve('polyumi_inference.dummy_server:app')

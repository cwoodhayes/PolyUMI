#!/usr/bin/env python3
"""
Push the end-effector payload to the FCI — runs ON THE NUC, once, at bringup.

A one-shot: call ``/service_server/set_load`` with what nuc/tcp_calib.py measures, check the
response, exit 0 or non-zero. fr3_bringup.launch.py runs it as an ExecuteProcess and refuses to
spawn fr3_arm_controller unless it exits 0 — so the exit status is what decides whether the arm
comes up at all, and it has to mean the payload actually took.

That is why this is a client rather than a `ros2 service call`: the CLI exits 0 whether the
response says ``success: true`` or ``success: false``, so gating on it means parsing its
human-readable output for "success=True". A repr-format change would then read a *successful*
SetLoad as a failure and refuse to bring up the robot. Reading ``response.success`` off the typed
message has no such contract.

Self-contained (no PolyUMI package deps) so it runs from a plain clone on the NUC:

    source /opt/ros/humble/setup.bash
    source ~/franka_ws/install/setup.bash
    python3 nuc/set_payload.py [timeout_s]
"""

from pathlib import Path
import sys

from franka_msgs.srv import SetLoad
import rclpy
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tcp_calib  # noqa: E402

SERVICE = '/service_server/set_load'
DEFAULT_TIMEOUT_S = 60.0


def _request() -> SetLoad.Request:
    """Fill a SetLoad request from tcp_calib, which owns every number in it."""
    req = SetLoad.Request()
    req.mass = float(tcp_calib.PAYLOAD_MASS)
    req.center_of_mass = [float(v) for v in tcp_calib.payload_com_flange()]
    req.load_inertia = [float(v) for v in tcp_calib.payload_inertia_flange()]
    return req


def set_payload(node: Node, timeout_s: float) -> str | None:
    """
    Call SetLoad once and report what went wrong, or None if it took.

    Split from :func:`main` so the failure paths are testable without a service on the other end.

    :param timeout_s: budget for the service appearing AND for the call itself. franka_bringup may
        still be starting, so waiting is normal; waiting forever is not, since bringup blocks on it.
    :returns: a human-readable reason, or None on success.
    """
    client = node.create_client(SetLoad, SERVICE)
    if not client.wait_for_service(timeout_sec=timeout_s):
        return f'{SERVICE} never appeared within {timeout_s:.0f}s — is franka_bringup up, and is FCI enabled?'

    future = client.call_async(_request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    if not future.done():
        return f'{SERVICE} did not answer within {timeout_s:.0f}s.'

    response = future.result()
    if response is None:
        return f'{SERVICE} call failed with no response.'
    if not response.success:
        # franka_param_service_server flattens every franka::CommandException to the string
        # "command exception error", so this rarely says anything useful on its own.
        return f'SetLoad was rejected: {response.error!r}'
    return None


def main(argv=None) -> int:
    """Set the payload; return a shell exit code."""
    argv = sys.argv[1:] if argv is None else argv
    timeout_s = float(argv[0]) if argv else DEFAULT_TIMEOUT_S

    rclpy.init()
    node = rclpy.create_node('set_payload')
    try:
        node.get_logger().info(f'[set_payload] {tcp_calib.describe_payload()}')
        reason = set_payload(node, timeout_s)
        if reason is not None:
            node.get_logger().error(f'[set_payload] {reason}')
            return 1
        node.get_logger().info('[set_payload] payload accepted by the FCI.')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())

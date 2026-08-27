#!/usr/bin/env python3
"""
Publish synthetic gripper chunks so franka_hand_node's dry run has a horizon to plan against.

Stands in for policy_client_node: same topic, same message, same absolute-schedule contract
(header.stamp + time_from_start). Run on the laptop with setup_franka_env.sh sourced, or on the
NUC directly. Nothing here moves anything -- launch the node with execute_gripper:=false.

    python3 gripper_chunk_stim.py                 # square wave, the pick-and-place shape
    python3 gripper_chunk_stim.py --shape ramp    # slow open/close, should stay branch A
    python3 gripper_chunk_stim.py --shape fast    # 3 Hz, faster than the hand can render

Re-publishes every 0.3 s at 10 Hz spacing over n_action_steps points, which is what the real
client does at steps_per_inference=3.
"""

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

CLOSED_M = 0.010
OPEN_M = 0.075


def widths(shape: str, t: float, n: int, dt: float) -> list[float]:
    """Width setpoints for one chunk starting at wall time t."""
    out = []
    for i in range(n):
        u = t + i * dt
        if shape == 'square':
            # 2 s closed, 2 s open. The shape that exposes "aim at the last horizon point",
            # which never closes the gripper at all.
            out.append(CLOSED_M if (u % 4.0) < 2.0 else OPEN_M)
        elif shape == 'ramp':
            out.append(CLOSED_M + (OPEN_M - CLOSED_M) * 0.5 * (1 - math.cos(2 * math.pi * u / 6.0)))
        else:  # fast -- past the hand's 1.38 Hz command Nyquist, so it must alias
            out.append(CLOSED_M + (OPEN_M - CLOSED_M) * 0.5 * (1 - math.cos(2 * math.pi * u * 3.0)))
    return out


def main():
    """Publish chunks until interrupted."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', choices=['square', 'ramp', 'fast'], default='square')
    ap.add_argument('--n', type=int, default=24, help='points per chunk (inference.yaml n_action_steps)')
    ap.add_argument('--dt', type=float, default=0.1, help='setpoint spacing (1/control_hz)')
    ap.add_argument('--topic', default='/polyumi/target_gripper')
    args = ap.parse_args()

    rclpy.init()
    node = Node('gripper_chunk_stim')
    pub = node.create_publisher(JointTrajectory, args.topic, 10)

    print(f'publishing {args.shape} chunks on {args.topic} ({args.n} pts @ {args.dt}s) -- ctrl-C to stop')
    t0 = time.time()
    try:
        while rclpy.ok():
            now = node.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()
            msg.joint_names = ['fr3_gripper_width']
            for i, w in enumerate(widths(args.shape, time.time() - t0, args.n, args.dt)):
                pt = JointTrajectoryPoint()
                pt.positions = [w]
                pt.time_from_start.sec = int(i * args.dt)
                pt.time_from_start.nanosec = int((i * args.dt % 1.0) * 1e9)
                msg.points.append(pt)
            pub.publish(msg)
            if pub.get_subscription_count() == 0:
                print('  WARNING: nothing subscribed -- is franka_hand_node running?')
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

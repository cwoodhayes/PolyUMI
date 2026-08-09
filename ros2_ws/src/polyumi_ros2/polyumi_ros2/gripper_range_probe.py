#!/usr/bin/env python3
"""
Measure the FR3 hand's real reachable aperture range, with the PolyUMI fingers fitted.

Gives the two robot-side numbers the gripper width map needs:

    A_closed  jaw aperture with the fingers touching  ->  pairs with S_closed to give the offset
    A_open    jaw aperture at full open               ->  gripper_max_width_m

Neither can be assumed. The Franka Hand's nominal stroke is 0–0.0817 m, but the custom fingers
collide well before the mechanism bottoms out and may also foul at the open end, so both limits are
properties of *these fingers* and have to be measured. `franka_gripper` does not publish max_width
on any topic either, so there is nothing to read back.

The offset then comes from pairing this with the ArUco side. "Fingers touching fingers" is the same
physical configuration on the handheld rig and on the arm, so with S_closed from
``pingest calibrate-gripper``::

    gripper_offset_m = S_closed - A_closed      # checkpoints trained on raw tag separation
    gripper_offset_m = -A_closed                # checkpoints trained after the exporter subtracts

Both are plausibly negative. That is fine and expected — see docs/franka-inference-bringup.md.

Why the spread matters as much as the mean: `Move` applies no force and stalls on contact, so the
closed endpoint is wherever the fingers happened to stop for that speed. If the spread across reps
is more than about a millimetre, the endpoint is not repeatable and you want it force-defined
instead — re-run with the bridge's ``use_grasp_below_m`` raised so a stated ``grasp_force_n``
decides where closed is. Don't reach for that pre-emptively; let the measurement say.

Usage (laptop, after `source setup_franka_env.sh`):

    # 1. NUC: the gripper must be allowed to move, or every command is a silent no-op
    ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true

    # 2. Nothing should be in the way of the fingers
    ros2 run polyumi_ros2 gripper_range_probe

**Do not run this while policy_client_node is running.** It publishes to /polyumi/target_gripper,
the same topic the policy uses, and the bridge acts on whichever chunk arrives last.
"""

import statistics
import threading
import time

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

#: Joint name the bridge expects on /polyumi/target_gripper (it reads positions, not names, but
#: matching policy_client_node keeps the topic self-describing).
GRIPPER_JOINT_NAME = 'fr3_gripper_width'

#: Give DDS time to match the bridge's subscription. Discovery is asynchronous, so a command
#: published before the endpoints pair goes nowhere, silently.
SUBSCRIBER_TIMEOUT_S = 10.0

#: Endpoint detection: the aperture is "settled" once it stops changing. Unlike the arm, the hand
#: makes short discrete moves, so there is no reversal to trip over — but there IS a delay before
#: the goal goes out (the bridge rate-limits to min_command_period_s) and a stationary hand looks
#: identical before and after. Hence a minimum wait, then a stability window.
COMMAND_LATENCY_S = 1.5
STABLE_TOL_M = 0.0002
STABLE_S = 1.0
SAMPLE_PERIOD_S = 0.05

#: Spread above which the endpoint is not repeatable enough to calibrate against.
REPEATABILITY_WARN_M = 0.001

#: If the open endpoint lands this close to what we commanded, we measured the bridge's clamp
#: rather than the hardware limit.
CLAMP_MARGIN_M = 0.001


class GripperRangeProbe(Node):
    """Drive the hand to both extremes repeatedly and report where it actually stops."""

    def __init__(self, **kwargs):
        """
        Declare params and set up the gripper publisher and state subscription.

        :param kwargs: forwarded to rclpy's Node — notably ``parameter_overrides``, matching the
            other scripts so this can be constructed under test without a launch file.
        """
        super().__init__('gripper_range_probe', **kwargs)

        self.declare_parameter('gripper_topic', '/polyumi/target_gripper')
        self.declare_parameter('gripper_state_topic', '/fr3_gripper/joint_states')
        # Commanded wide. The bridge clamps this to its own max_width_m (0.08 by default), which
        # is itself a guess — if the measured open endpoint lands at the clamp we say so, because
        # then the number is the software limit and not the fingers'.
        self.declare_parameter('open_width_m', 0.09)
        self.declare_parameter('closed_width_m', 0.0)
        self.declare_parameter('reps', 3)
        self.declare_parameter('settle_timeout_s', 15.0)

        self._open_width = self.get_parameter('open_width_m').get_parameter_value().double_value
        self._closed_width = self.get_parameter('closed_width_m').get_parameter_value().double_value
        self._reps = self.get_parameter('reps').get_parameter_value().integer_value
        self._settle_timeout_s = (
            self.get_parameter('settle_timeout_s').get_parameter_value().double_value
        )
        topic = self.get_parameter('gripper_topic').get_parameter_value().string_value
        state_topic = self.get_parameter('gripper_state_topic').get_parameter_value().string_value

        if self._reps < 1:
            raise ValueError(f'reps must be >= 1, got {self._reps}')
        if self._open_width <= self._closed_width:
            raise ValueError(
                f'open_width_m ({self._open_width}) must exceed closed_width_m ({self._closed_width})'
            )

        self._state_topic = state_topic
        self._pub = self.create_publisher(JointTrajectory, topic, 10)
        self._lock = threading.Lock()
        self._aperture: float | None = None
        self.create_subscription(JointState, state_topic, self._on_state, 10)

    def _on_state(self, msg: JointState) -> None:
        """Cache the aperture; each FR3 finger reports half of it."""
        if len(msg.position) >= 2:
            with self._lock:
                self._aperture = float(msg.position[0] + msg.position[1])

    def aperture(self) -> float | None:
        """Return the most recent jaw aperture in metres, or None if nothing has arrived."""
        with self._lock:
            return self._aperture

    def wait_for_state(self, timeout_s: float = 10.0) -> bool:
        """Block until the hand is reporting, so a missing gripper fails here and not mid-sweep."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.aperture() is not None:
                return True
            time.sleep(0.1)
        self.get_logger().error(
            f'Nothing published on {self._state_topic} after {timeout_s:.0f}s — is fr3_bringup up '
            'on the NUC with load_gripper:=true (the default)?'
        )
        return False

    def wait_for_subscriber(self, timeout_s: float = SUBSCRIBER_TIMEOUT_S) -> bool:
        """
        Block until the bridge has matched our publisher, so the first command is not dropped.

        Publishing into an unmatched topic loses the message with no error anywhere, which would
        surface later as "the gripper never moved" and send you looking at execute_gripper.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._pub.get_subscription_count() > 0:
                return True
            time.sleep(0.1)
        self.get_logger().error(
            f'Nothing is subscribed to {self._pub.topic_name} after {timeout_s:.0f}s — is '
            'fr3_gripper_bridge running on the NUC (ros2 launch nuc/launch/fr3_inference.launch.py)?'
        )
        return False

    def command(self, width_m: float) -> None:
        """Publish a single-waypoint gripper chunk; the bridge turns it into a Move/Grasp goal."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [GRIPPER_JOINT_NAME]
        point = JointTrajectoryPoint()
        point.positions = [width_m]
        # The bridge derives its move speed from this; ~1 s of allotted travel keeps the approach
        # gentle, which matters at the closed end where the fingers stall against each other.
        point.time_from_start = Duration(seconds=1.0).to_msg()
        msg.points.append(point)
        self._pub.publish(msg)

    def settle(self) -> float | None:
        """
        Wait for the aperture to stop changing and return where it stopped.

        Motion is NOT required first: commanding open when the hand is already open legitimately
        moves nothing, and the bridge's deadband may swallow the goal outright. Requiring movement
        would hang on exactly that case, so this waits out the command latency and then looks for
        stability.
        """
        time.sleep(COMMAND_LATENCY_S)
        deadline = time.monotonic() + self._settle_timeout_s
        last = self.aperture()
        stable_since = None
        while time.monotonic() < deadline:
            time.sleep(SAMPLE_PERIOD_S)
            current = self.aperture()
            if current is None:
                continue
            if last is not None and abs(current - last) <= STABLE_TOL_M:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since > STABLE_S:
                    return current
            else:
                stable_since = None
            last = current
        self.get_logger().warning(f'Aperture never settled within {self._settle_timeout_s:.0f}s.')
        return self.aperture()

    def _endpoint(self, width_m: float, label: str) -> float | None:
        """Command one extreme and report where the hand actually came to rest."""
        self.command(width_m)
        measured = self.settle()
        if measured is None:
            self.get_logger().error(f'{label}: no aperture reading.')
            return None
        self.get_logger().info(f'{label}: commanded {width_m:.4f}m, settled at {measured:.4f}m')
        return measured

    def run(self) -> int:
        """Drive both extremes `reps` times and print the calibration numbers."""
        if not self.wait_for_state() or not self.wait_for_subscriber():
            return 1

        self.get_logger().warning(
            'MOVING THE FINGERS through their full range — make sure nothing is between them.'
        )

        opens: list[float] = []
        closeds: list[float] = []
        for i in range(self._reps):
            self.get_logger().info(f'--- rep {i + 1}/{self._reps} ---')
            measured_open = self._endpoint(self._open_width, 'open ')
            measured_closed = self._endpoint(self._closed_width, 'closed')
            if measured_open is None or measured_closed is None:
                return 1
            opens.append(measured_open)
            closeds.append(measured_closed)

        return self._report(opens, closeds)

    def _report(self, opens: list[float], closeds: list[float]) -> int:
        """Print the endpoints with their spreads, and the config lines they feed."""
        a_open, spread_open = statistics.fmean(opens), max(opens) - min(opens)
        a_closed, spread_closed = statistics.fmean(closeds), max(closeds) - min(closeds)

        self.get_logger().info(
            f'A_open   = {a_open:.4f} m  (spread {spread_open * 1000:.2f} mm over {len(opens)} reps)'
        )
        self.get_logger().info(
            f'A_closed = {a_closed:.4f} m  (spread {spread_closed * 1000:.2f} mm over {len(closeds)} reps)'
        )
        self.get_logger().info(f'usable stroke = {(a_open - a_closed) * 1000:.1f} mm')

        ok = True
        if spread_closed > REPEATABILITY_WARN_M:
            ok = False
            self.get_logger().error(
                f'The closed endpoint varies by {spread_closed * 1000:.2f} mm between reps, so it '
                'is not repeatable enough to calibrate against. Move applies no force and stalls '
                'wherever it happens to, so make the endpoint force-defined instead: re-run with '
                'the bridge started at use_grasp_below_m:=0.02 (and a chosen grasp_force_n).'
            )
        if spread_open > REPEATABILITY_WARN_M:
            ok = False
            self.get_logger().error(
                f'The open endpoint varies by {spread_open * 1000:.2f} mm between reps — unexpected '
                'for a free move, so suspect something fouling the fingers.'
            )
        if abs(a_open - self._open_width) > CLAMP_MARGIN_M and a_open < self._open_width:
            # Landing short of the command is the normal case (the fingers or the bridge's clamp
            # stop it first); saying which is what makes the number actionable.
            self.get_logger().warning(
                f'Open settled {(self._open_width - a_open) * 1000:.1f} mm short of the command. If '
                "that is the bridge's max_width_m rather than the fingers, raise it and re-run — "
                'otherwise this IS the fingers, which is the number you want.'
            )

        self.get_logger().info(
            'Put in ros2_ws/src/polyumi_ros2/config/inference.yaml:\n'
            f'    gripper_max_width_m: {a_open:.4f}\n'
            f'    gripper_min_width_m: {a_closed:.4f}\n'
            '  and combine A_closed with S_closed from `pingest calibrate-gripper` for '
            'gripper_offset_m (see the module docstring for which formula).'
        )
        return 0 if ok else 1


def main():
    """Spin the node on a background thread and run the probe in the foreground."""
    rclpy.init()
    try:
        node = GripperRangeProbe()
    except ValueError as e:
        print(f'gripper_range_probe: {e}')
        rclpy.shutdown()
        return 2
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        code = node.run()
    except KeyboardInterrupt:
        code = 130
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == '__main__':
    raise SystemExit(main())

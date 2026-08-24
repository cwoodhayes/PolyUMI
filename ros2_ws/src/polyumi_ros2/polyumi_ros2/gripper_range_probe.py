#!/usr/bin/env python3
"""
Measure the FR3 hand's real reachable aperture range, with the PolyUMI fingers fitted.

Gives the two robot-side numbers the gripper width map needs, both of them jaw apertures read off
``/fr3_gripper/joint_states``:

    the **closed aperture**  with the fingers touching  ->  ``gripper_min_width_m``, which doubles
                                                            as the policy->robot offset
    the **open aperture**    at full open               ->  ``gripper_max_width_m``

Neither can be assumed. The Franka Hand's nominal stroke is 0–0.0817 m, but the custom fingers can
collide before the mechanism bottoms out and can foul at the open end, so both limits are properties
of *these fingers*. `franka_gripper` does not publish max_width on any topic either, so there is
nothing to read back.

Measured 2026-08-09: the closed aperture is **0.0000 m** and the open aperture **0.0816 m**. These
fingers meet exactly at the mechanism's zero rather than colliding early — which matches how
``gripper_calib.yaml`` defines the fingertip frame ("the point where the two fingertips meet" when
fully closed) — and they do not foul at the open end either, so both limits are the hand's own
rather than the fingers'. The first run reported 0.0800 because the bridge's default clamp stopped
it, which is why this probe now *asks* the bridge for its clamp instead of guessing.

Cross-check against the ArUco side, which is independent of everything here: the **closed width**
(``gripper_calib.yaml``'s ``closed_mm``, the tag separation with the fingers touching) is 44.56 mm,
so the FR3 at full open corresponds to 44.56 + 81.6 = 126.2 mm of tag separation, against the
132.3 mm the handheld actually reaches. The handheld therefore opens 6.2 mm wider — about 7% of the
policy's commanded range saturates at the top, which is expected and shows up as intent clipping
rather than an error.

Both go straight into ``config/inference.yaml`` as ``gripper_min_width_m`` and
``gripper_max_width_m``. There is no third constant to derive: the closed aperture *is* the
policy->robot offset, since policy width 0 means fully closed and fully closed on the arm is that
aperture. (A separate ``gripper_offset_m`` used to exist and was removed — it was always exactly
the negated closed aperture, so it duplicated this measurement.) See polyumi_ros2.gripper_map.

Why the spread matters as much as the mean: `Move` applies no force and stalls on contact, so the
closed endpoint is wherever the fingers happened to stop for that speed. If the spread across reps
is more than about a millimetre, the endpoint is not repeatable and you want it force-defined
instead — re-run with the bridge's ``use_grasp_below_m`` raised so a stated ``grasp_force_n``
decides where closed is. Don't reach for that pre-emptively; let the measurement say.

Usage (laptop, after `source setup_franka_env.sh`):

    # 1. NUC: the gripper must be allowed to move, or every command is a silent no-op. The
    #    bridge's clamp already defaults to 0.0817 m — the Franka Hand's own maximum — so the
    #    fingers stop the open sweep first if anything does. Pass gripper_max_width only if you
    #    have deliberately lowered it; a clamp below the hand's maximum measures the software.
    ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true

    # 2. Nothing should be in the way of the fingers
    ros2 run polyumi_ros2 gripper_range_probe

**Do not run this while policy_client_node is running.** It publishes to /polyumi/target_gripper,
the same topic the policy uses, and the bridge acts on whichever chunk arrives last.
"""

import statistics
import threading
import time

from rcl_interfaces.srv import GetParameters
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

#: How close the open endpoint has to sit to the bridge's clamp before we call it clamped rather
#: than a real stop.
CLAMP_MARGIN_M = 0.001

#: franka_hand_node's parameter service (it is named `fr3_gripper`, inheriting franka_gripper's
#: node name so /fr3_gripper/joint_states keeps working). Queried rather than guessed: "did I measure the fingers or the
#: software limit" is otherwise undecidable from this side, and guessing it from the spread gets it
#: wrong — a hand that reaches its own maximum every time is just as repeatable as one hitting a
#: clamp. Service calls DO cross the Humble<->Kilted rmw gap even though the ROS graph does not, so
#: this works from the laptop; see docs/crb-fr3-inference.md.
BRIDGE_PARAM_SERVICE = '/fr3_gripper/get_parameters'
BRIDGE_PARAM_TIMEOUT_S = 5.0

#: The Franka Hand's own maximum aperture after homing. Once the clamp is here there is nothing
#: further to try — franka_gripper ABORTS widths past max_width rather than clamping them — so
#: "settled at the clamp" stops being a software artifact and becomes the hardware answer. Without
#: this the probe would keep telling you to raise a clamp that cannot usefully go higher.
HAND_MAX_WIDTH_M = 0.0817


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
        # Commanded past the hand's maximum on purpose, so the fingers decide where open is. The
        # bridge clamps it to its own max_width_m, which the inference launch file defaults to the
        # hand's 0.0817 m — if the measured endpoint lands on a clamp *below* that we say so,
        # because then the number is a software limit and not the fingers'.
        self.declare_parameter('open_width_m', 0.09)
        self.declare_parameter('closed_width_m', 0.0)
        self.declare_parameter('reps', 3)
        self.declare_parameter('settle_timeout_s', 15.0)

        self._open_width = self.get_parameter('open_width_m').get_parameter_value().double_value
        self._closed_width = self.get_parameter('closed_width_m').get_parameter_value().double_value
        self._reps = self.get_parameter('reps').get_parameter_value().integer_value
        self._settle_timeout_s = self.get_parameter('settle_timeout_s').get_parameter_value().double_value
        topic = self.get_parameter('gripper_topic').get_parameter_value().string_value
        state_topic = self.get_parameter('gripper_state_topic').get_parameter_value().string_value

        # Node.__init__ has already registered us with the rclpy context, so tear that down before
        # raising — otherwise every rejected construction leaves an orphan node behind.
        errors = []
        if self._reps < 1:
            errors.append(f'reps must be >= 1, got {self._reps}')
        if self._open_width <= self._closed_width:
            errors.append(f'open_width_m ({self._open_width}) must exceed closed_width_m ({self._closed_width})')
        if errors:
            self.destroy_node()
            raise ValueError('; '.join(errors))

        self._state_topic = state_topic
        self._pub = self.create_publisher(JointTrajectory, topic, 10)
        self._bridge_params = self.create_client(GetParameters, BRIDGE_PARAM_SERVICE)
        self._lock = threading.Lock()
        self._aperture: float | None = None
        self.create_subscription(JointState, state_topic, self._on_state, 10)

    def bridge_max_width(self) -> float | None:
        """
        Ask the bridge what it clamps commanded widths to, or None if it cannot be reached.

        Without this the probe cannot distinguish "the fingers stopped here" from "the software
        stopped here", and the two demand opposite responses: one is the number you want, the
        other means re-run with a higher clamp.
        """
        if not self._bridge_params.wait_for_service(timeout_sec=BRIDGE_PARAM_TIMEOUT_S):
            self.get_logger().warning(
                f'{BRIDGE_PARAM_SERVICE} did not answer; cannot tell whether the open endpoint is '
                "the fingers or the bridge's clamp."
            )
            return None
        request = GetParameters.Request()
        request.names = ['max_width_m']
        future = self._bridge_params.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=BRIDGE_PARAM_TIMEOUT_S):
            return None
        response = future.result()
        if response is None or not response.values:
            return None
        return float(response.values[0].double_value)

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
            'franka_hand_node running on the NUC (ros2 launch nuc/launch/fr3_inference.launch.py)?'
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

        self.get_logger().warning('MOVING THE FINGERS through their full range — make sure nothing is between them.')

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

        return self._report(opens, closeds, clamp_m=self.bridge_max_width())

    def _report(self, opens: list[float], closeds: list[float], clamp_m: float | None = None) -> int:
        """Print the endpoints with their spreads, and the config lines they feed."""
        a_open, spread_open = statistics.fmean(opens), max(opens) - min(opens)
        a_closed, spread_closed = statistics.fmean(closeds), max(closeds) - min(closeds)

        self.get_logger().info(
            f'open aperture   = {a_open:.4f} m  (spread {spread_open * 1000:.2f} mm over {len(opens)} reps)'
        )
        self.get_logger().info(
            f'closed aperture = {a_closed:.4f} m  (spread {spread_closed * 1000:.2f} mm over {len(closeds)} reps)'
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
        if clamp_m is None:
            # Not fatal: the endpoints are still measured, they are just unverified against the
            # software limit. Say so rather than inventing a verdict.
            self.get_logger().warning(
                f"Could not read the bridge's max_width_m, so open aperture = {a_open:.4f} m might be the "
                'software clamp rather than a physical stop. Check it by hand before using this.'
            )
        elif a_open < clamp_m - CLAMP_MARGIN_M:
            self.get_logger().info(
                f'open aperture = {a_open:.4f} m stopped {(clamp_m - a_open) * 1000:.1f} mm below the '
                f"bridge's clamp ({clamp_m:.4f} m), so hardware stopped it, not software. This is "
                'the number you want.'
            )
        elif clamp_m >= HAND_MAX_WIDTH_M - CLAMP_MARGIN_M:
            self.get_logger().info(
                f'open aperture = {a_open:.4f} m sits at the clamp ({clamp_m:.4f} m), but that clamp is '
                f"already the Franka Hand's own maximum ({HAND_MAX_WIDTH_M:.4f} m) and "
                'franka_gripper aborts anything wider — so there is nothing further to command and '
                'this IS the hardware limit. The number you want.'
            )
        else:
            ok = False
            self.get_logger().error(
                f"open aperture = {a_open:.4f} m is the bridge's max_width_m clamp ({clamp_m:.4f} m), not "
                'a physical stop — the hand was never allowed to open further, so this measures '
                'the software limit. Re-run with a higher clamp:\n'
                '    ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true '
                f'gripper_max_width:={HAND_MAX_WIDTH_M}\n'
                "  (that is the Franka Hand's own maximum; do not go higher, franka_gripper aborts "
                'widths past it rather than clamping.)'
            )

        self.get_logger().info(
            'Put in ros2_ws/src/polyumi_ros2/config/inference.yaml:\n'
            f'    gripper_max_width_m: {a_open:.4f}\n'
            f'    gripper_min_width_m: {a_closed:.4f}\n'
            '  No offset constant to set: gripper_min_width_m is the offset. Cross-check '
            'against `pingest calibrate-gripper` (closed width + open aperture should land near '
            'the handheld open_mm).'
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

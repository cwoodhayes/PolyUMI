#!/usr/bin/env python3
"""
FR3 gripper bridge — runs ON THE NUC (ROS 2 Humble).

Subscribes to the gripper half of the inference action chunk on /polyumi/target_gripper (a
trajectory_msgs/JointTrajectory published by the laptop's policy_client_node over DDS, one point
per action step, positions in metres of JAW APERTURE) and drives the Franka Hand via its local
action servers. Runs on the NUC for the same reason as fr3_moveit_bridge: same-rmw as the hardware.

WHY THIS IS NOT A SERVO. UMI drives its gripper with a 30 Hz interpolated position+velocity
stream (umi/real_world/wsg_controller.py, the same PoseTrajectoryInterpolator its arm uses). The
Franka Hand cannot do that at any level of the stack: it has NO ros2_control interface at all
(`ros2 control list_hardware_interfaces` lists nothing for the fingers), and libfranka's
franka::Gripper exposes only blocking homing/grasp/move/stop. So instead of tracking every
waypoint, this node holds the latest desired width and issues a discrete goal only when the
target has moved past a deadband and a minimum period has elapsed.

That rate limit is mandatory, not a nicety: franka_gripper's action server ACCEPTs every goal
unconditionally into a detached thread, with no queue and no preemption, and libfranka aborts
whichever command a new goal supersedes. Commanding at the 10 Hz control rate would produce ~9
aborted goals per second plus unbounded thread churn.

Self-contained (no PolyUMI package deps) so it runs from a plain clone on the NUC:
    source /opt/ros/humble/setup.bash
    source ~/franka_ws/install/setup.bash   # franka_gripper must be up (fr3-bringup)
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file://$HOME/franka_ws/config/cyclonedds.xml
    python3 nuc/fr3_gripper_bridge.py --ros-args -p execute:=false   # dry-run (no motion)

Set execute:=true to actually move the fingers. Default is false (log only) for safety.
"""

import threading
import time

from franka_msgs.action import Grasp, Move
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

DEFAULT_TARGET_TOPIC = '/polyumi/target_gripper'
DEFAULT_STATE_TOPIC = '/fr3_gripper/joint_states'

# The hand's true max is ~0.0817 m after homing, but max_width is not published on any topic —
# it lives only inside franka_gripper — so it has to be a constant here.
DEFAULT_MAX_WIDTH = 0.08
# Watchdog on the single in-flight goal slot. Goal resolution is callback-driven, so a goal that
# never resolves (server died mid-motion, result callback lost) would otherwise wedge the bridge
# permanently — it would stop commanding and never say why. A full-stroke Move at our slowest
# allowed speed takes a few seconds, so this is generous.
GOAL_RESULT_TIMEOUT_S = 10.0


class Fr3GripperBridge(Node):
    """Drive the Franka Hand from inference gripper-width chunks, deadbanded and rate-limited."""

    def __init__(self):
        """Declare params, create the target subscription and gripper action clients."""
        super().__init__('fr3_gripper_bridge')

        self.declare_parameter('execute', False)
        self.declare_parameter('target_topic', DEFAULT_TARGET_TOPIC)
        self.declare_parameter('state_topic', DEFAULT_STATE_TOPIC)
        # Which point of the chunk to track. A chunk spans ~0.8 s while the hand needs 100s of ms
        # to traverse, so 0 (track the first action) inherently lags; a small lead anticipates the
        # policy's intent at the cost of closing early. Tune on hardware.
        self.declare_parameter('gripper_lead_steps', 0)
        # Ignore commanded widths within this of the last one we actually sent. This is what keeps
        # a jittering policy output from turning into a goal storm.
        self.declare_parameter('width_deadband_m', 0.005)
        # Never issue goals faster than this, regardless of deadband.
        self.declare_parameter('min_command_period_s', 0.25)
        self.declare_parameter('max_width_m', DEFAULT_MAX_WIDTH)
        # Move speed is derived from the chunk's own timing (see _desired_speed); these bound it.
        self.declare_parameter('min_speed_mps', 0.02)
        self.declare_parameter('max_speed_mps', 0.15)
        # Grasp (force-controlled) instead of Move when closing below this width. 0.0 disables it,
        # which is the shipped default: Move applies no force, so bringup moves fingers and nothing
        # more. Raise it once you actually want to hold objects.
        self.declare_parameter('use_grasp_below_m', 0.0)
        self.declare_parameter('grasp_force_n', 20.0)
        # Deliberately wide (the franka_gripper default is 0.005): with a continuously-commanded
        # width, a tight band reports failure whenever the object isn't exactly the commanded size.
        # Wide epsilon means "close until you hit something, then squeeze".
        self.declare_parameter('grasp_epsilon_m', 0.05)

        self._execute = self.get_parameter('execute').get_parameter_value().bool_value
        topic = self.get_parameter('target_topic').get_parameter_value().string_value
        state_topic = self.get_parameter('state_topic').get_parameter_value().string_value
        self._lead_steps = self.get_parameter('gripper_lead_steps').get_parameter_value().integer_value
        self._deadband = self.get_parameter('width_deadband_m').get_parameter_value().double_value
        self._period = self.get_parameter('min_command_period_s').get_parameter_value().double_value
        self._max_width = self.get_parameter('max_width_m').get_parameter_value().double_value
        self._min_speed = self.get_parameter('min_speed_mps').get_parameter_value().double_value
        self._max_speed = self.get_parameter('max_speed_mps').get_parameter_value().double_value
        self._grasp_below = self.get_parameter('use_grasp_below_m').get_parameter_value().double_value
        self._grasp_force = self.get_parameter('grasp_force_n').get_parameter_value().double_value
        self._grasp_eps = self.get_parameter('grasp_epsilon_m').get_parameter_value().double_value

        self._cbgroup = ReentrantCallbackGroup()
        self._move = ActionClient(self, Move, '/fr3_gripper/move', callback_group=self._cbgroup)
        self._grasp = ActionClient(self, Grasp, '/fr3_gripper/grasp', callback_group=self._cbgroup)

        # Latest-wins target; no queue. _last_commanded is what we actually sent (not what was
        # last requested), so the deadband measures drift from the hand's commanded state.
        self._state_lock = threading.Lock()
        self._desired_width: float | None = None
        self._desired_lead_s = self._period
        self._last_commanded: float | None = None
        self._current_width: float | None = None
        self._goal_in_flight = False
        self._goal_sent_at: float | None = None

        self.create_subscription(JointTrajectory, topic, self._on_target, 10, callback_group=self._cbgroup)
        self.create_subscription(JointState, state_topic, self._on_state, 10, callback_group=self._cbgroup)
        self.create_timer(self._period, self._tick, callback_group=self._cbgroup)

        # Fail loudly at startup rather than on the first chunk.
        for client, name in ((self._move, 'move'), (self._grasp, 'grasp')):
            if not client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error(
                    f'/fr3_gripper/{name} action server NOT found after 10s — is fr3-bringup running '
                    'on this NUC, and was it started with load_gripper:=true (the default)?'
                )
            else:
                self.get_logger().info(f'/fr3_gripper/{name} ready.')

        mode = 'EXECUTE (fingers will move)' if self._execute else 'log-only (no motion)'
        self.get_logger().info(f'fr3_gripper_bridge started — listening on {topic} — mode: {mode}')
        self.get_logger().info(
            f'rate limiting — deadband={self._deadband}m, min period={self._period}s, '
            f'lead={self._lead_steps} steps, speed in [{self._min_speed}, {self._max_speed}] m/s'
        )
        grasp_mode = (
            f'Grasp below {self._grasp_below}m ({self._grasp_force}N, eps={self._grasp_eps}m)'
            if self._grasp_below > 0
            else 'Move only — position control, applies NO force, will not hold an object'
        )
        self.get_logger().info(f'action selection — {grasp_mode}')

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _on_state(self, msg: JointState) -> None:
        """Cache the current aperture (each FR3 finger reports half of it)."""
        if len(msg.position) < 2:
            return
        with self._state_lock:
            self._current_width = float(msg.position[0] + msg.position[1])

    def _on_target(self, msg: JointTrajectory) -> None:
        """Record the latest desired width from a chunk; the timer decides whether to send it."""
        if not msg.points:
            self.get_logger().warning('Received empty gripper chunk, ignoring.')
            return
        idx = min(max(self._lead_steps, 0), len(msg.points) - 1)
        point = msg.points[idx]
        if not point.positions:
            self.get_logger().warning('Gripper chunk point has no positions, ignoring.')
            return
        width = min(max(float(point.positions[0]), 0.0), self._max_width)
        # How long the policy expects to take getting there — used to pick a move speed so the
        # hand travels on roughly the intended timeline instead of always at one fixed rate.
        lead_s = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
        with self._state_lock:
            self._desired_width = width
            self._desired_lead_s = max(lead_s, self._period)

    # ------------------------------------------------------------------
    # Command loop
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Issue at most one gripper goal per period, and only if the target has really moved."""
        now = time.monotonic()
        with self._state_lock:
            desired = self._desired_width
            lead_s = self._desired_lead_s
            current = self._current_width
            last = self._last_commanded
            in_flight = self._goal_in_flight
            sent_at = self._goal_sent_at
        if in_flight:
            # Watchdog: a goal whose result never arrives would otherwise block every future
            # command silently. Free the slot and say so.
            if sent_at is not None and now - sent_at > GOAL_RESULT_TIMEOUT_S:
                self.get_logger().warning(
                    f'Gripper goal has been in flight for {now - sent_at:.1f}s with no result; '
                    'releasing the slot. If this repeats, check that fr3-bringup is still up.'
                )
                self._clear_in_flight()
            return
        if desired is None:
            return
        if last is not None and abs(desired - last) < self._deadband:
            return

        speed = self._desired_speed(desired, current, lead_s)
        use_grasp = (
            self._grasp_below > 0.0
            and desired < self._grasp_below
            and current is not None
            and desired < current
        )
        action = 'Grasp' if use_grasp else 'Move'
        if not self._execute:
            self.get_logger().info(
                f'[log-only] would send {action}(width={desired:.4f}m, speed={speed:.3f}m/s)'
            )
            with self._state_lock:
                self._last_commanded = desired
            return

        with self._state_lock:
            self._goal_in_flight = True
            self._goal_sent_at = now
            self._last_commanded = desired
        self.get_logger().info(f'{action}(width={desired:.4f}m, speed={speed:.3f}m/s)')
        if use_grasp:
            goal = Grasp.Goal()
            goal.width = desired
            goal.speed = speed
            goal.force = self._grasp_force
            goal.epsilon.inner = self._grasp_eps
            goal.epsilon.outer = self._grasp_eps
            self._send(self._grasp, goal, action)
        else:
            goal = Move.Goal()
            goal.width = desired
            goal.speed = speed
            self._send(self._move, goal, action)

    def _desired_speed(self, desired: float, current: float | None, lead_s: float) -> float:
        """
        Pick a move speed so the hand traverses roughly on the chunk's own timeline.

        The closest discrete analogue to UMI's velocity-aware PD stream: we cannot command a
        velocity profile, but we can at least size the single speed to the distance and the time
        the policy allotted for it, instead of always moving at one fixed rate.
        """
        if current is None:
            return self._max_speed
        return min(max(abs(desired - current) / lead_s, self._min_speed), self._max_speed)

    def _send(self, client: ActionClient, goal, label: str) -> None:
        """Send a gripper goal and free the in-flight slot when it resolves."""
        if not client.server_is_ready():
            self.get_logger().error(f'{label} server not available; dropping goal.')
            self._clear_in_flight()
            return
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_goal_response(f, label))

    def _on_goal_response(self, future, label: str) -> None:
        """Handle goal acceptance, then chain to the result."""
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warning(f'{label} goal rejected.')
            self._clear_in_flight()
            return
        handle.get_result_async().add_done_callback(lambda f: self._on_goal_result(f, label))

    def _on_goal_result(self, future, label: str) -> None:
        """
        Log the outcome and free the slot.

        A failed/aborted result is logged at INFO, not as an error: franka_gripper has no
        preemption, so libfranka aborts whichever command the next goal supersedes, and a
        moving target legitimately produces these. Treating them as faults would generate
        exactly the log spam the rate limiting exists to prevent. A *steady stream* of them
        still means the deadband or period is too small — see docs/crb-fr3-inference.md.
        """
        result = future.result()
        if result is None:
            self.get_logger().warning(f'{label} returned no result.')
        elif not result.result.success:
            self.get_logger().info(f'{label} did not complete: {result.result.error}')
        self._clear_in_flight()

    def _clear_in_flight(self) -> None:
        """Release the single-goal slot so the next tick can command again."""
        with self._state_lock:
            self._goal_in_flight = False
            self._goal_sent_at = None


def main():
    """Spin the bridge node under a multithreaded executor."""
    rclpy.init()
    node = Fr3GripperBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

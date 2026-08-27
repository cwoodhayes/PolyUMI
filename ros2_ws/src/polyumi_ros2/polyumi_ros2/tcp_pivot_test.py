#!/usr/bin/env python3
"""
Pivot test: check whether ``polyumi_tcp`` really sits at the fingertips.

Commands a **pure rotation about the TCP** — the frame's origin is held fixed while its
orientation sweeps — and you watch the closed fingertips. If the frame is right, the fingertips
stay put and the arm swings around them. If the frame is off by a vector ``d``, they trace an arc
of radius ``|d|``. Nothing in software can check this for you: TF will always report the TCP
exactly where the URDF says it is, so the model-versus-reality gap is only visible in the room.

Reading the result — each axis exposes the two error components perpendicular to it, so running
all three localises the error completely:

    rotate about TCP x  ->  reveals error in y and z
    rotate about TCP y  ->  reveals error in x and z
    rotate about TCP z  ->  reveals error in x and y   (z is the approach axis)

A mirrored ``Rz`` sign in nuc/tcp_calib.py shows up here as the fingertips sweeping ~15 cm rather
than holding still.

What it DOES check automatically: that the executed motion really held the TCP still *according
to the robot model*. It samples TF throughout and reports the peak deviation. A few mm is normal
tracking error; centimetres mean the plan or the bridge is not controlling polyumi_tcp at all,
which is a plumbing bug rather than a calibration one. That distinction is the point — it tells
you whether to go looking in the launch files or in the CAD.

Usage (laptop, after `source setup_franka_env.sh`):

    # 1. NUC: bringup + inference. Both execute flags on — this closes the hand itself — and
    #    SLOW. execute_gripper:=false makes the close a silent no-op, which the script detects.
    ros2 launch nuc/launch/fr3_inference.launch.py \
        execute_arm:=true execute_gripper:=true

    # 2. Get to a known, roomy pose — a pivot near the workspace edge will fail to plan
    ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"

    # 3. Watch the fingertips. The gripper is closed automatically first.
    ros2 run polyumi_ros2 tcp_pivot_test --ros-args -p angle_deg:=20.0

**Do not run this while policy_client_node is running.** It publishes to
/polyumi/target_poses_traj, the same topic the policy uses, and the controller splices whichever
chunk arrives last.
"""

import math
import threading
import time

from geometry_msgs.msg import Pose
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from polyumi_ros2.target_chunk import CONSUMER_HINT, TargetChunkPublisher
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Waypoints are handed over explicitly at this spacing rather than as two endpoints: the
# controller interpolates between whatever it is given, and a pure rotation commanded as one
# 20-degree step would be taken as a single jump.
DEFAULT_STEP_DEG = 5.0
# Completion is detected as "it moved, then stopped", in angular RATE rather than per-sample
# delta. A raw delta threshold is a trap: at these waypoint rates a sample is a few thousandths
# of a radian, so any plausible epsilon sits on top of the real signal — and the sweep REVERSES at
# both extremes, where the arm genuinely passes through zero velocity mid-motion. Requiring
# sustained quiet, after motion has been seen, survives both.
MOTION_RATE_RAD_S = 0.01
QUIET_S = 2.0
SAMPLE_PERIOD_S = 0.05
# The rate is measured against a sample this old, NOT against the previous one. At vscale=0.05 the
# sweep's real rate is only ~0.03 rad/s, so an adjacent-sample delta (0.05 s of travel) sits close
# enough to the arm's own jitter that noise can latch `moved` and then keep resetting the quiet
# timer — the sweep would never be seen to finish and would burn the whole sweep_timeout_s.
# A longer baseline divides the noise by the window while leaving the real signal untouched.
RATE_WINDOW_S = 0.5
# Give DDS time to match the bridge's subscription. Discovery is asynchronous, so a chunk
# published before the endpoints pair goes nowhere, silently.
SUBSCRIBER_TIMEOUT_S = 10.0


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions (a then b, i.e. b applied in a's frame)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians between two xyzw orientations (double-cover safe)."""
    dot = min(abs(float(np.dot(a, b))), 1.0)
    return 2.0 * math.acos(dot)


def axis_quat(axis: str, angle_rad: float) -> np.ndarray:
    """Build an xyzw quaternion rotating by angle_rad about a principal axis."""
    half = angle_rad / 2.0
    s = math.sin(half)
    unit = {'x': (s, 0.0, 0.0), 'y': (0.0, s, 0.0), 'z': (0.0, 0.0, s)}[axis]
    return np.array([*unit, math.cos(half)])


def sweep_angles(angle_deg: float, step_deg: float) -> list[float]:
    """
    Angles for a 0 -> +A -> -A -> 0 sweep, in radians, at roughly step_deg spacing.

    The leading 0 is a literal first element, not just the implicit starting point: the timed wire
    stamps waypoint 0 at "now", which is already stale by the time it reaches the NUC, so whichever
    angle is first in this list is the one silently dropped. Making that angle 0 (the pose the arm
    already holds) means dropping it costs nothing; without it, the real first step (angle 1) was
    the one that got dropped, and the sweep's first VISIBLE motion was two steps wide instead of one.
    """
    # ceil, not round: rounding down would space the waypoints WIDER than asked (30deg at 7deg
    # spacing rounds to 4 steps of 7.5), and that spacing is the interpolation resolution.
    steps = max(math.ceil(abs(angle_deg) / max(step_deg, 1e-6)), 1)
    out = [0.0]  # the pose already held, so losing it as the stale first waypoint is free
    out += [angle_deg * i / steps for i in range(1, steps + 1)]  # 0 -> +A
    out += [angle_deg * (1 - 2 * i / (2 * steps)) for i in range(1, 2 * steps + 1)]  # +A -> -A
    out += [-angle_deg * (1 - i / steps) for i in range(1, steps + 1)]  # -A -> 0
    return [math.radians(a) for a in out]


class TcpPivotTest(Node):
    """Command pure rotations about the TCP and report how far the TCP itself drifted."""

    def __init__(self, **kwargs):
        """
        Declare params and set up the TF listener and target-pose publisher.

        :param kwargs: forwarded to rclpy's Node — notably ``parameter_overrides``, matching the
            NUC bridges so this can be constructed under test without a launch file.
        """
        super().__init__('tcp_pivot_test', **kwargs)

        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('eef_frame', 'polyumi_tcp')
        self.declare_parameter('target_topic', '')
        # Seconds between waypoints. Slow on purpose: this test is meant to be watched, and it
        # also bounds how far the equilibrium point leads the arm.
        self.declare_parameter('waypoint_dt_s', 0.5)
        self.declare_parameter('angle_deg', 20.0)
        self.declare_parameter('step_deg', DEFAULT_STEP_DEG)
        # Which TCP axes to pivot about, in order. See the module docstring for what each reveals.
        self.declare_parameter('axes', 'xyz')
        self.declare_parameter('sweep_timeout_s', 90.0)
        self.declare_parameter('settle_s', 2.0)
        # How long to allow for plan + execution start before concluding nothing is going to move.
        self.declare_parameter('motion_start_timeout_s', 20.0)
        # Ceiling on how far the TCP may drift and still be called a pure rotation. Generous,
        # because some of it is legitimate: the streaming controller's interpolator pins the TCP
        # only AT the waypoints, and joint-space interpolation between two 5-degree-apart
        # orientations bows the TCP out by a few mm. Tighten step_deg before tightening this.
        self.declare_parameter('max_drift_mm', 25.0)
        # Close the hand first — a pivot test with open fingers has no fingertip to watch. Goes
        # out on the bridge's own topic rather than the NUC action servers, since that path is
        # proven to cross the rmw gap. Requires execute_gripper:=true on the NUC.
        self.declare_parameter('close_gripper', True)
        self.declare_parameter('gripper_topic', '/polyumi/target_gripper')
        self.declare_parameter('gripper_state_topic', '/fr3_gripper/joint_states')
        self.declare_parameter('gripper_width_m', 0.0)
        self.declare_parameter('gripper_close_timeout_s', 15.0)

        self._base = self.get_parameter('base_frame').get_parameter_value().string_value
        self._eef = self.get_parameter('eef_frame').get_parameter_value().string_value
        topic = self.get_parameter('target_topic').get_parameter_value().string_value or None
        self._waypoint_dt = self.get_parameter('waypoint_dt_s').get_parameter_value().double_value
        self._angle_deg = self.get_parameter('angle_deg').get_parameter_value().double_value
        self._step_deg = self.get_parameter('step_deg').get_parameter_value().double_value
        self._axes = self.get_parameter('axes').get_parameter_value().string_value
        self._timeout_s = self.get_parameter('sweep_timeout_s').get_parameter_value().double_value
        self._settle_s = self.get_parameter('settle_s').get_parameter_value().double_value
        self._motion_start_timeout_s = self.get_parameter('motion_start_timeout_s').get_parameter_value().double_value
        self._max_drift_m = self.get_parameter('max_drift_mm').get_parameter_value().double_value / 1000.0
        self._close_gripper = self.get_parameter('close_gripper').get_parameter_value().bool_value
        gripper_topic = self.get_parameter('gripper_topic').get_parameter_value().string_value
        gripper_state = self.get_parameter('gripper_state_topic').get_parameter_value().string_value
        self._gripper_width = self.get_parameter('gripper_width_m').get_parameter_value().double_value
        self._gripper_timeout_s = self.get_parameter('gripper_close_timeout_s').get_parameter_value().double_value

        bad = [a for a in self._axes if a not in 'xyz']
        if bad or not self._axes:
            raise ValueError(f"axes must be a non-empty string over 'xyz', got {self._axes!r}")
        if self._step_deg <= 0:
            raise ValueError(f'step_deg must be > 0, got {self._step_deg}')
        if self._waypoint_dt <= 0:
            raise ValueError(f'waypoint_dt_s must be > 0, got {self._waypoint_dt}')

        self._pub = TargetChunkPublisher(self, frame_id=self._base, joint_name=self._eef, topic=topic)
        self._gripper_pub = self.create_publisher(JointTrajectory, gripper_topic, 10)
        self._gripper_lock = threading.Lock()
        self._gripper_actual: float | None = None
        self.create_subscription(JointState, gripper_state, self._on_gripper_state, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def _on_gripper_state(self, msg: JointState) -> None:
        """Cache the aperture; each FR3 finger reports half of it."""
        if len(msg.position) >= 2:
            with self._gripper_lock:
                self._gripper_actual = float(msg.position[0] + msg.position[1])

    def close_gripper(self) -> bool:
        """
        Command the hand shut and wait until it reports closed.

        Verified rather than assumed: the whole test is "watch the fingertips", so open fingers
        make the result meaningless, and a silently-ignored close (execute_gripper:=false) is
        otherwise invisible from here.
        """
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ['fr3_gripper_width']
        point = JointTrajectoryPoint()
        point.positions = [self._gripper_width]
        # The bridge sizes its move speed from this; ~1 s of allotted travel closes gently
        # rather than at the max_speed_mps a zero would ask for.
        point.time_from_start = Duration(seconds=1.0).to_msg()
        msg.points.append(point)
        self._gripper_pub.publish(msg)
        self.get_logger().info(f'Closing the gripper to {self._gripper_width:.3f}m...')

        deadline = time.monotonic() + self._gripper_timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.1)
            with self._gripper_lock:
                actual = self._gripper_actual
            # Deliberately loose: Move applies no force and stalls on contact, so the fingers
            # meeting each other lands near, not at, the commanded width.
            if actual is not None and actual <= self._gripper_width + 0.006:
                self.get_logger().info(f'Gripper closed (width {actual:.4f}m).')
                return True
        with self._gripper_lock:
            actual = self._gripper_actual
        seen = 'no /fr3_gripper/joint_states at all' if actual is None else f'width {actual:.4f}m'
        self.get_logger().error(
            f'Gripper did not close within {self._gripper_timeout_s}s ({seen}). Is the NUC '
            'inference stack running with execute_gripper:=true? Re-run with '
            '-p close_gripper:=false to skip this and close it yourself.'
        )
        return False

    def lookup(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (position, xyzw quaternion) of eef_frame in base_frame, or None."""
        try:
            tf = self._tf_buffer.lookup_transform(self._base, self._eef, rclpy.time.Time())
        except Exception:  # noqa: BLE001 — any TF failure is just "not ready yet" to the caller
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w])

    def wait_for_tf(self, timeout_s: float = 15.0) -> tuple[np.ndarray, np.ndarray] | None:
        """Block until the TCP transform is available, or give up."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pose = self.lookup()
            if pose is not None:
                return pose
            time.sleep(0.1)
        return None

    def wait_for_subscriber(self, pub, consumer: str, timeout_s: float = SUBSCRIBER_TIMEOUT_S) -> bool:
        """
        Block until `pub` has matched a subscriber, so the first message is not silently dropped.

        Publishing into an unmatched topic loses the message with no error anywhere. That would
        surface much later as "the arm never moved" (or a gripper that never closed), whose
        diagnosis blames execute_arm or an in-flight chunk — sending you to the wrong pane.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if pub.get_subscription_count() > 0:
                return True
            time.sleep(0.1)
        self.get_logger().error(f'Nothing is subscribed to {pub.topic_name} after {timeout_s:.0f}s. Needs: {consumer}')
        return False

    def _publish_sweep(self, axis: str, position: np.ndarray, start_quat: np.ndarray) -> int:
        """Publish one chunk holding `position` while rotating about the TCP's own `axis`."""
        poses = []
        for angle in sweep_angles(self._angle_deg, self._step_deg):
            # Right-multiply: the delta is applied in the TCP's OWN frame, which is what makes
            # this a rotation about the TCP's axes rather than about the base frame's.
            q = quat_mul(start_quat, axis_quat(axis, angle))
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in q)
            poses.append(pose)
        self._pub.publish(poses, dt=self._waypoint_dt)
        return len(poses)

    def _watch(self, position: np.ndarray) -> tuple[float, bool]:
        """
        Sample TF until the arm has moved and then stopped.

        :returns: (peak TCP drift from `position` in metres, whether any motion was seen).

        The bridge publishes no status, so motion end has to be observed rather than awaited.
        Requiring motion BEFORE quiet counts is the important half: without it, the plan latency
        and the sweep's own direction reversals both read as "finished", and the next chunk goes
        out while the arm is still moving — where the bridge drops it as "previous plan/execute
        in flight" and the run silently measures nothing.
        """
        start = time.monotonic()
        max_drift = 0.0
        ref_quat, ref_t = None, None
        moved = False
        quiet_since = None
        while time.monotonic() - start < self._timeout_s:
            time.sleep(SAMPLE_PERIOD_S)
            pose = self.lookup()
            if pose is None:
                continue
            current, quat = pose
            now = time.monotonic()
            # Drift is sampled every period; the rate is only evaluated once the baseline is
            # RATE_WINDOW_S old, so intermediate samples are recorded but leave the baseline alone.
            max_drift = max(max_drift, float(np.linalg.norm(current - position)))
            if ref_quat is None:
                ref_quat, ref_t = quat, now
            elif now - ref_t >= RATE_WINDOW_S:
                rate = quat_angle(ref_quat, quat) / (now - ref_t)
                ref_quat, ref_t = quat, now
                if rate > MOTION_RATE_RAD_S:
                    moved = True
                    quiet_since = None
                elif moved:
                    quiet_since = quiet_since or now
                    if now - quiet_since > QUIET_S:
                        return max_drift, True
            if not moved and now - start > self._motion_start_timeout_s:
                return max_drift, False
        self.get_logger().warning(f'Sweep did not settle within {self._timeout_s}s.')
        return max_drift, moved

    def run(self) -> int:
        """Run one sweep per requested axis; return a shell exit code."""
        start = self.wait_for_tf()
        if start is None:
            self.get_logger().error(
                f'No TF {self._base} -> {self._eef}. Is fr3_bringup up on the NUC, and did you '
                'source setup_franka_env.sh?'
            )
            return 1
        if not self.wait_for_subscriber(self._pub, CONSUMER_HINT):
            return 1
        if self._close_gripper:
            if not self.wait_for_subscriber(
                self._gripper_pub,
                'franka_hand_node (ros2 launch nuc/launch/fr3_inference.launch.py on the NUC)',
            ):
                return 1
            if not self.close_gripper():
                return 1
        position, quat = start
        self.get_logger().info(
            f'{self._eef} starts at [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] '
            f'in {self._base}. Pivoting +/-{self._angle_deg}deg about axes "{self._axes}".'
        )
        self.get_logger().warning(
            'WATCH THE FINGERTIPS. They should stay put while the arm swings around them; '
            'an arc means polyumi_tcp is offset from the real fingertips by the arc radius.'
        )

        drifts = {}
        for axis in self._axes:
            n = self._publish_sweep(axis, position, quat)
            self.get_logger().info(f'--- pivoting about TCP {axis} ({n} waypoints) ---')
            drift, moved = self._watch(position)
            if not moved:
                # Reporting a drift here would be a lie — it would be noise measured on a
                # stationary arm, and it would look like a pass.
                self.get_logger().error(
                    f'axis {axis}: the arm never moved. Either the NUC is running with '
                    'execute_arm:=false, or the bridge dropped this chunk ("previous '
                    'plan/execute in flight") because the last sweep was still running. Check '
                    'the fr3-inference pane.'
                )
                return 1
            drifts[axis] = drift
            self.get_logger().info(f'axis {axis}: peak model TCP drift {drift * 1000:.1f} mm')
            time.sleep(self._settle_s)

        worst = max(drifts.values())
        self.get_logger().info(
            'Model-side result (does NOT validate the calibration — only that the robot did what '
            'was asked): peak TCP drift ' + ', '.join(f'{a}={d * 1000:.1f}mm' for a, d in drifts.items())
        )
        if worst > self._max_drift_m:
            self.get_logger().error(
                f'Peak drift {worst * 1000:.1f} mm exceeds max_drift_mm. The executed motion was '
                'not a pure rotation about the TCP. Most likely the waypoints are too coarse for '
                'the planner to hold the TCP between them — try a smaller step_deg. If that does '
                'not help, check move_group loaded fr3_polyumi.urdf.xacro and eef_link is '
                'polyumi_tcp.'
            )
            return 1
        self.get_logger().info('The robot held the TCP still. Whether the FINGERTIPS held still is yours to judge.')
        return 0


def main():
    """Spin the node on a background thread and run the sweeps in the foreground."""
    rclpy.init()
    try:
        # Inside the try: the parameter validation in __init__ raises, and letting that escape
        # would skip rclpy.shutdown() and bury the message under a traceback.
        node = TcpPivotTest()
    except ValueError as exc:
        print(f'tcp_pivot_test: {exc}')
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

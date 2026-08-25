"""
Drive the streaming Cartesian impedance controller with a synthetic trajectory, no policy.

**This moves the arm.** It is the step between "the arm springs back when pushed" and "run the
policy": the first proves the control law, this proves the *reference generator* — that overlapping
chunks arriving at a policy-like rate splice into continuous motion.

Nothing else covers that. ``tcp_pivot_test`` publishes one chunk per sweep and waits for it to
finish, and ``latency_probe --mode arm`` publishes single waypoints, so neither ever has two
multi-waypoint chunks in flight at once. That overlap is the whole point of the interpolator, and it
is where a timing sign error shows up as a stutter or a jerk rather than as a wrong number.

The path is a circle traced around wherever the TCP already is, at a speed you can watch.
Orientation is held: a wrong orientation would be a second variable in a test that exists to isolate
one.

    # NUC: fr3_bringup up, impedance controller ACTIVE (docs/crb-fr3-inference.md)
    ros2 run polyumi_ros2 servo_smoke_test
    ros2 run polyumi_ros2 servo_smoke_test --ros-args -p radius_m:=0.05 -p period_s:=6.0

The plain invocation is the one validated on hardware (2026-08-19) — do not run a bigger or faster
override as the FIRST pass on a new arm.

Watch for smooth continuous motion, no pause at chunk boundaries, and no ``cartesian_reflex``. A
stutter at exactly ``chunk_hz`` is the splice going wrong, not the gains.

Do NOT run this while ``policy_client_node`` is up — they publish to the same topic and the
controller acts on whichever chunk arrived last.
"""

import math
import threading
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from polyumi_ros2.target_chunk import TargetChunkPublisher, Wire

#: How long to wait for the first TF sample before giving up.
TF_TIMEOUT_S = 10.0
#: How long to wait for an executor to subscribe. Discovery is asynchronous.
SUBSCRIBER_TIMEOUT_S = 10.0


class ServoSmokeTest(Node):
    """Publish overlapping, absolutely-timed pose chunks tracing a circle around the current TCP."""

    def __init__(self, **kwargs):
        """
        Declare parameters and set up TF plus the chunk publisher.

        :param kwargs: forwarded to Node — notably ``parameter_overrides``, so this is constructible
            under test without a launch file.
        :raises ValueError: on any parameter that would make the test meaningless rather than wrong.
        """
        super().__init__('servo_smoke_test', **kwargs)

        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('eef_frame', 'polyumi_tcp')
        self.declare_parameter('target_topic', '')
        self.declare_parameter('radius_m', 0.03)
        # Seconds for one lap. Slow by default: this is meant to be watched, not timed.
        self.declare_parameter('period_s', 12.0)
        self.declare_parameter('duration_s', 30.0)
        # Plane of the circle, as two distinct base-frame axes.
        self.declare_parameter('plane', 'xy')
        # These three make the traffic policy-shaped. Defaults mirror the real inference loop: a
        # 16-waypoint chunk at 0.1 s spacing re-issued every ~0.3 s, so each chunk supersedes the
        # tail of the previous one with over a second still unplayed.
        self.declare_parameter('chunk_hz', 3.3)
        self.declare_parameter('waypoints_per_chunk', 16)
        self.declare_parameter('waypoint_dt_s', 0.1)
        # Subtracted from each chunk anchor, exactly as policy_client_node applies latency.arm_exec.
        # Left at 0 so this test exercises the splice rather than the latency model.
        #
        # NOT latency_probe's `lead_s`, which is ADDED to schedule waypoints into the future. Here a
        # larger value moves every waypoint earlier, so more of each chunk is dropped as stale —
        # the opposite effect. Hence the different name.
        self.declare_parameter('arm_exec_s', 0.0)

        self._base = self.get_parameter('base_frame').get_parameter_value().string_value
        self._eef = self.get_parameter('eef_frame').get_parameter_value().string_value
        topic = self.get_parameter('target_topic').get_parameter_value().string_value or None
        self._radius = self.get_parameter('radius_m').get_parameter_value().double_value
        self._period = self.get_parameter('period_s').get_parameter_value().double_value
        self._duration = self.get_parameter('duration_s').get_parameter_value().double_value
        self._plane = self.get_parameter('plane').get_parameter_value().string_value
        self._chunk_hz = self.get_parameter('chunk_hz').get_parameter_value().double_value
        self._n_waypoints = self.get_parameter('waypoints_per_chunk').get_parameter_value().integer_value
        self._waypoint_dt = self.get_parameter('waypoint_dt_s').get_parameter_value().double_value
        self._arm_exec = self.get_parameter('arm_exec_s').get_parameter_value().double_value

        self._validate()

        self._pub = TargetChunkPublisher(
            self, wire=Wire.MULTIDOF, frame_id=self._base, joint_name=self._eef, topic=topic
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def _validate(self) -> None:
        """
        Reject configurations that would run happily and test nothing.

        :raises ValueError: with every problem at once, so a bad invocation is fixed in one pass.
        """
        errors = []
        if self._radius <= 0:
            errors.append(f'radius_m must be > 0, got {self._radius}')
        if self._period <= 0:
            errors.append(f'period_s must be > 0, got {self._period}')
        if self._chunk_hz <= 0:
            errors.append(f'chunk_hz must be > 0, got {self._chunk_hz}')
        if self._waypoint_dt <= 0:
            errors.append(f'waypoint_dt_s must be > 0, got {self._waypoint_dt}')
        if self._n_waypoints < 2:
            errors.append(f'waypoints_per_chunk must be >= 2, got {self._n_waypoints}')
        if len(self._plane) != 2 or any(a not in 'xyz' for a in self._plane) or self._plane[0] == self._plane[1]:
            errors.append(f"plane must be two distinct axes from 'xyz', got {self._plane!r}")

        # Overlap is the entire point. Chunks that finish before the next arrives make this a slow
        # sequence of independent moves, which proves nothing tcp_pivot_test does not already.
        # Span is (n-1)*dt, not n*dt: n waypoints at spacing dt cover n-1 intervals, from the first
        # to the last. Overcounting by one dt let configurations through whose chunks do not
        # actually overlap — the exact thing this check exists to catch.
        if self._chunk_hz > 0 and self._waypoint_dt > 0 and self._n_waypoints >= 2:
            span = (self._n_waypoints - 1) * self._waypoint_dt
            interval = 1.0 / self._chunk_hz
            if span <= interval:
                errors.append(
                    f'chunk span ({span:.2f}s) must exceed the chunk interval ({interval:.2f}s), '
                    'or chunks never overlap and the splice is not exercised'
                )
        if errors:
            raise ValueError('; '.join(errors))

    def wait_for_tf(self) -> Pose | None:
        """Return the current TCP pose, or None if TF never arrives."""
        deadline = time.monotonic() + TF_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                tf = self._tf_buffer.lookup_transform(self._base, self._eef, rclpy.time.Time())
            except Exception:  # noqa: BLE001 - tf2 raises several unrelated types while warming up
                time.sleep(0.1)
                continue
            pose = Pose()
            pose.position.x = tf.transform.translation.x
            pose.position.y = tf.transform.translation.y
            pose.position.z = tf.transform.translation.z
            pose.orientation = tf.transform.rotation
            return pose
        self.get_logger().error(
            f'No {self._base} -> {self._eef} transform after {TF_TIMEOUT_S:.0f}s — is fr3_bringup running?'
        )
        return None

    def wait_for_subscriber(self) -> bool:
        """Block until an executor subscribes; on timeout name who was expected."""
        deadline = time.monotonic() + SUBSCRIBER_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._pub.get_subscription_count() > 0:
                return True
            time.sleep(0.1)
        self.get_logger().error(
            f'Nothing is subscribed to {self._pub.topic_name} after {SUBSCRIBER_TIMEOUT_S:.0f}s. '
            f'Needs: {self._pub.wire.consumer}'
        )
        return False

    def pose_at(self, centre: Pose, phase: float) -> Pose:
        """
        Build the circle's pose at `phase` radians, holding the centre's orientation.

        Offset so that phase 0 is the centre itself rather than one radius away — the first waypoint
        published is then where the arm already is, and the motion starts without a step.
        """
        pose = Pose()
        pose.position.x = centre.position.x
        pose.position.y = centre.position.y
        pose.position.z = centre.position.z
        first, second = self._plane
        setattr(
            pose.position,
            first,
            getattr(centre.position, first) + self._radius * (math.cos(phase) - 1.0),
        )
        setattr(pose.position, second, getattr(centre.position, second) + self._radius * math.sin(phase))
        pose.orientation = centre.orientation
        return pose

    def chunk_at(self, centre: Pose, elapsed: float) -> list[Pose]:
        """Waypoints for the chunk issued at `elapsed` seconds, spaced by waypoint_dt_s."""
        return [
            self.pose_at(centre, 2 * math.pi * (elapsed + i * self._waypoint_dt) / self._period)
            for i in range(self._n_waypoints)
        ]

    def run(self) -> bool:
        """Trace the circle for `duration_s`. False if it could not start."""
        centre = self.wait_for_tf()
        if centre is None:
            return False
        if not self.wait_for_subscriber():
            return False

        span = (self._n_waypoints - 1) * self._waypoint_dt
        interval = 1.0 / self._chunk_hz
        self.get_logger().warn(
            f'MOVING THE ARM: {self._radius * 100:.1f} cm circle in {self._plane}, '
            f'{self._period:.0f}s per lap, for {self._duration:.0f}s. Chunks: {self._n_waypoints} '
            f'waypoints spanning {span:.2f}s, every {interval:.2f}s ({span / interval:.1f}x overlap).'
        )

        start = self.get_clock().now()
        while (elapsed := (self.get_clock().now() - start).nanoseconds * 1e-9) < self._duration:
            anchor = self.get_clock().now() - Duration(seconds=self._arm_exec)
            self._pub.publish(self.chunk_at(centre, elapsed), dt=self._waypoint_dt, stamp=anchor.to_msg())
            time.sleep(interval)

        self.get_logger().info('Done. The arm holds its last commanded pose.')
        return True


def main():
    """Run the smoke test, spinning the node on a background thread so TF fills."""
    rclpy.init()
    node = ServoSmokeTest()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        ok = node.run()
    except KeyboardInterrupt:
        ok = False
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()

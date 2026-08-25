"""
Tests for the TCP pivot test's geometry.

The whole validity of the pivot test rests on rotating about the TCP's OWN axes. Left-multiplying
the delta instead would pivot about the base frame, which still looks like a plausible arm motion
on hardware while telling you nothing about where the fingertips are — a false negative you would
believe. The rest is sweep bookkeeping, which is easy to get subtly wrong (missing the endpoints,
steps larger than requested) and impossible to notice on the robot.
"""

import math

import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter

from polyumi_ros2.tcp_pivot_test import (
    TcpPivotTest,
    axis_quat,
    quat_angle,
    quat_mul,
    sweep_angles,
)


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    """Init/shutdown rclpy once for the whole module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def _matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_delta_rotates_about_the_tcps_own_axis(axis):
    """
    The composed rotation must be about the TCP's axis expressed in base coordinates.

    Verified by rotating a start orientation that is deliberately NOT identity: with q0 = identity
    the body-frame and base-frame compositions agree, so the bug this guards against would hide.
    """
    q0 = axis_quat('x', math.pi / 2)  # TCP is tilted 90 deg from the base frame
    r0 = _matrix(q0)
    expected_axis = r0 @ {'x': np.array([1.0, 0, 0]), 'y': np.array([0, 1.0, 0]), 'z': np.array([0, 0, 1.0])}[axis]

    composed = _matrix(quat_mul(q0, axis_quat(axis, math.radians(30))))
    # A rotation leaves its own axis unchanged; the composed frame's motion relative to the start
    # is `composed @ r0.T`, whose fixed direction must be the TCP axis in base coordinates.
    relative = composed @ r0.T
    assert relative @ expected_axis == pytest.approx(expected_axis, abs=1e-9)


def test_left_multiplying_would_pivot_about_the_base_frame():
    """Pin the distinction this module depends on, so a 'harmless' reorder gets caught."""
    q0 = axis_quat('x', math.pi / 2)
    delta = axis_quat('z', math.radians(30))
    assert not np.allclose(quat_mul(q0, delta), quat_mul(delta, q0))


def test_sweep_visits_both_extremes_and_returns_to_start():
    """0 -> +A -> -A -> 0, so the arm ends where it began and the test is repeatable."""
    angles = np.degrees(sweep_angles(20.0, 5.0))

    assert angles[0] == pytest.approx(0.0), 'leading 0 must be a literal element, not implicit'
    assert angles[-1] == pytest.approx(0.0)
    assert angles.max() == pytest.approx(20.0)
    assert angles.min() == pytest.approx(-20.0)


def test_sweep_steps_never_exceed_the_requested_spacing():
    """
    Keep the waypoint spacing at or under what was asked for.

    GetCartesianPath's max_step bounds translation only, so a pure rotation is never subdivided
    by the planner — the spacing here IS the interpolation resolution.
    """
    for angle, step in ((20.0, 5.0), (30.0, 7.0), (5.0, 1.0)):
        # sweep_angles' own first element is now the leading 0, so the list already includes that
        # first step; no need to prepend one here as well.
        angles = np.degrees(sweep_angles(angle, step))
        deltas = np.abs(np.diff(angles))
        assert deltas.max() <= step + 1e-9, f'{angle}deg at {step}deg spacing'


def _capture_publisher(node):
    """Swap the chunk publisher for a recorder, returning the list of (poses, kwargs) it sees."""
    published = []

    class _Recorder:
        def publish(self, poses, **kwargs):
            published.append((poses, kwargs))

    node._pub = _Recorder()
    return published


def test_published_sweep_holds_the_position_fixed():
    """Any position variation across the chunk makes it not a pivot; only orientation may change."""
    node = TcpPivotTest(parameter_overrides=[Parameter('angle_deg', value=20.0)])
    published = _capture_publisher(node)
    try:
        position = np.array([0.4, -0.1, 0.5])
        n = node._publish_sweep('z', position, axis_quat('x', math.pi / 2))

        poses, _ = published[0]
        assert n == len(poses) > 1
        for pose in poses:
            assert (pose.position.x, pose.position.y, pose.position.z) == pytest.approx(position)
        orientations = {(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w) for p in poses}
        assert len(orientations) > 1, 'the chunk must actually rotate'
    finally:
        node.destroy_node()


def test_sweep_carries_the_waypoint_spacing():
    """
    The timed wire format needs a dt or every waypoint lands at the same instant.

    A chunk whose waypoints all share one timestamp is not a slow pivot — it is a step to the last
    pose, which is exactly the discontinuity the interpolator exists to prevent.
    """
    node = TcpPivotTest(parameter_overrides=[Parameter('waypoint_dt_s', value=0.25)])
    published = _capture_publisher(node)
    try:
        node._publish_sweep('z', np.array([0.4, 0.0, 0.5]), axis_quat('x', 0.0))

        _, kwargs = published[0]
        assert kwargs['dt'] == pytest.approx(0.25)
    finally:
        node.destroy_node()


@pytest.mark.parametrize('bad', [{'axes': ''}, {'axes': 'xq'}, {'step_deg': 0.0}])
def test_invalid_parameters_fail_fast(bad):
    """A bad axis string would otherwise silently sweep nothing and 'pass'."""
    params = [Parameter(k, value=v) for k, v in bad.items()]
    with pytest.raises(ValueError):
        TcpPivotTest(parameter_overrides=params)


@pytest.mark.parametrize('degrees', [0.0, 15.0, 90.0, 179.0])
def test_quat_angle_measures_the_rotation_between_orientations(degrees):
    """Completion detection is a rate in rad/s, so this has to be a real angle, not a norm."""
    q0 = axis_quat('y', math.radians(33.0))  # arbitrary, non-identity
    q1 = quat_mul(q0, axis_quat('z', math.radians(degrees)))

    assert math.degrees(quat_angle(q0, q1)) == pytest.approx(degrees, abs=1e-6)


def test_quat_angle_ignores_quaternion_double_cover():
    """Treat q and -q as the same orientation; a 180 deg reading there would fake motion."""
    q = axis_quat('x', math.radians(40.0))

    assert quat_angle(q, -q) == pytest.approx(0.0, abs=1e-6)


def _gripper_node(**overrides):
    """Build a node whose gripper publisher records instead of publishing."""
    params = [Parameter(k, value=v) for k, v in overrides.items()]
    node = TcpPivotTest(parameter_overrides=params)
    sent = []
    node._gripper_pub = type('P', (), {'publish': lambda _s, msg: sent.append(msg)})()
    return node, sent


def test_close_gripper_commands_a_single_closed_waypoint():
    """The bridge reads points[0].positions[0]; anything else silently commands the wrong width."""
    node, sent = _gripper_node(gripper_width_m=0.0)
    node._gripper_actual = 0.0  # already closed, so the wait returns immediately
    try:
        assert node.close_gripper()
    finally:
        node.destroy_node()

    assert len(sent) == 1
    assert len(sent[0].points) == 1
    assert list(sent[0].points[0].positions) == [0.0]
    assert list(sent[0].joint_names) == ['fr3_gripper_width']
    # A zero here would make the bridge derive its maximum move speed; see close_gripper.
    assert sent[0].points[0].time_from_start.sec == 1


def test_close_gripper_fails_when_the_hand_never_reports_closed():
    """
    execute_gripper:=false makes the close a silent no-op, invalidating the whole run.

    The fingertips would be apart, so 'do they stay put' would be measuring nothing in
    particular — the failure has to be loud rather than a warning nobody reads.
    """
    node, _ = _gripper_node(gripper_close_timeout_s=0.3)
    node._gripper_actual = 0.078  # wide open, and never changes
    try:
        assert not node.close_gripper()
    finally:
        node.destroy_node()


def test_close_gripper_tolerates_fingers_stalling_just_short():
    """Move applies no force and stalls on contact, so 'closed' lands near, not at, the target."""
    node, _ = _gripper_node(gripper_width_m=0.0, gripper_close_timeout_s=0.3)
    node._gripper_actual = 0.004
    try:
        assert node.close_gripper()
    finally:
        node.destroy_node()


class _Pub:
    """A publisher stub whose subscription count appears after `after` polls."""

    topic_name = '/polyumi/target_poses'

    def __init__(self, after: int):
        self._after = after
        self.polls = 0

    def get_subscription_count(self) -> int:
        self.polls += 1
        return 1 if self.polls > self._after else 0


def test_wait_for_subscriber_returns_once_the_bridge_matches():
    """DDS discovery is asynchronous, so the wait has to poll rather than sample once."""
    node = TcpPivotTest(parameter_overrides=[])
    try:
        assert node.wait_for_subscriber(_Pub(after=2), 'fr3_moveit_bridge', timeout_s=5.0)
    finally:
        node.destroy_node()


def test_wait_for_subscriber_gives_up_when_nothing_is_listening():
    """
    A chunk published into an unmatched topic vanishes with no error anywhere.

    That must fail here, naming the bridge — not later as 'the arm never moved', which sends you
    to look at execute_arm instead of at discovery.
    """
    node = TcpPivotTest(parameter_overrides=[])
    try:
        assert not node.wait_for_subscriber(_Pub(after=10**6), 'fr3_moveit_bridge', timeout_s=0.3)
    finally:
        node.destroy_node()


def test_rate_baseline_is_older_than_one_sample():
    """
    The motion threshold is only meaningful against a window, not adjacent samples.

    At vscale=0.05 the sweep turns ~0.03 rad/s; over one 0.05 s sample that is 1.5 mrad, close
    enough to arm jitter that noise latches `moved` and starves the quiet detector.
    """
    from polyumi_ros2.tcp_pivot_test import MOTION_RATE_RAD_S, RATE_WINDOW_S, SAMPLE_PERIOD_S

    assert RATE_WINDOW_S >= 5 * SAMPLE_PERIOD_S
    # The smallest rotation the window can resolve must stay well under one waypoint of the sweep.
    assert MOTION_RATE_RAD_S * RATE_WINDOW_S < math.radians(1.0)

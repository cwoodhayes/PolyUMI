"""
Tests for the servo smoke test's trajectory geometry and its configuration guards.

This node exists to prove the interpolator splices overlapping chunks smoothly, so the two things
worth pinning are that the path it publishes is actually continuous and starts where the arm
already is, and that a configuration which quietly stops testing the splice is rejected rather than
run. Both would otherwise look fine on hardware and tell you nothing.
"""

import math

from geometry_msgs.msg import Pose
import pytest
import rclpy
from rclpy.parameter import Parameter

from polyumi_ros2.servo_smoke_test import ServoSmokeTest
from polyumi_ros2.target_chunk import Wire


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    """Init/shutdown rclpy once for the whole module."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def make_node():
    """Construct ServoSmokeTest with parameter overrides, destroying it afterwards."""
    nodes = []

    def _make(**params):
        node = ServoSmokeTest(parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node.destroy_node()


def _centre() -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = 0.4, -0.1, 0.5
    pose.orientation.w = 1.0
    return pose


def test_first_waypoint_is_where_the_arm_already_is(make_node):
    """
    Phase 0 must be the centre, not one radius away.

    The controller's equilibrium is seeded at the measured pose, so a chunk whose first waypoint sits
    a radius away commands a step the moment this starts — the exact discontinuity the interpolator
    exists to avoid, delivered by the tool meant to prove it works.
    """
    node = make_node(radius_m=0.05)
    centre = _centre()

    start = node.pose_at(centre, 0.0)

    assert (start.position.x, start.position.y, start.position.z) == pytest.approx(
        (centre.position.x, centre.position.y, centre.position.z)
    )


def test_path_is_a_circle_of_the_requested_radius(make_node):
    """A quarter turn must be one radius off in each axis of the plane; otherwise it is not a circle."""
    node = make_node(radius_m=0.05, plane='xy')
    centre = _centre()

    quarter = node.pose_at(centre, math.pi / 2)

    # cos(pi/2) - 1 = -1, sin(pi/2) = +1, both scaled by the radius.
    assert quarter.position.x == pytest.approx(centre.position.x - 0.05)
    assert quarter.position.y == pytest.approx(centre.position.y + 0.05)
    assert quarter.position.z == pytest.approx(centre.position.z), 'the off-plane axis must not move'


def test_orientation_is_held(make_node):
    """Rotation would be a second variable in a test that exists to isolate translation tracking."""
    node = make_node()
    centre = _centre()

    for phase in (0.0, 1.0, math.pi, 5.0):
        pose = node.pose_at(centre, phase)
        assert pose.orientation == centre.orientation


def test_consecutive_chunks_overlap_in_time_and_agree_where_they_cross(make_node):
    """
    The splice only means anything if a new chunk restates poses the previous one had not reached.

    Where two chunks describe the same instant they must agree, or the interpolator is being asked
    to reconcile two different intended paths and the resulting motion is not the tool's fault.
    """
    # 2 Hz rather than the shipped 3.3 so the interval is a whole number of waypoints (0.5s = 5 x
    # 0.1s) and the two chunks land on the same instants. At 3.3 the shift is 3.03 waypoints, so
    # they interleave instead of coinciding and the comparison below has nothing exact to check.
    # The overlap ratio is 3.2x either way, so the property under test is unchanged.
    node = make_node(waypoints_per_chunk=16, waypoint_dt_s=0.1, chunk_hz=2.0, period_s=12.0)
    centre = _centre()
    interval = 0.5

    first = node.chunk_at(centre, 0.0)
    second = node.chunk_at(centre, interval)

    # The chunks overlap: the second starts partway into the first.
    shift = round(interval / 0.1)
    assert 0 < shift < len(first)
    for i in range(len(first) - shift):
        assert second[i].position.x == pytest.approx(first[i + shift].position.x, abs=1e-9)
        assert second[i].position.y == pytest.approx(first[i + shift].position.y, abs=1e-9)


def test_non_overlapping_chunks_are_rejected(make_node):
    """
    A chunk that finishes before the next arrives makes this a sequence of separate moves.

    That configuration runs perfectly happily and proves nothing tcp_pivot_test does not already,
    which is precisely why it has to fail loudly rather than quietly.
    """
    with pytest.raises(ValueError, match='overlap'):
        make_node(waypoints_per_chunk=2, waypoint_dt_s=0.05, chunk_hz=1.0)


@pytest.mark.parametrize(
    'bad',
    [
        {'radius_m': 0.0},
        {'period_s': 0.0},
        {'plane': 'xx'},
        {'plane': 'xq'},
        {'plane': 'xyz'},
        {'waypoints_per_chunk': 1},
    ],
)
def test_invalid_parameters_fail_fast(make_node, bad):
    """Each of these would otherwise move the arm in a way that looks deliberate."""
    with pytest.raises(ValueError):
        make_node(**bad)


def test_only_drives_the_streaming_controller(make_node):
    """There is no PoseArray form of this test — MoveIt cannot splice, so the concept does not apply."""
    node = make_node()

    assert node._pub.wire is Wire.MULTIDOF
    assert node._pub.topic_name == '/polyumi/target_poses_traj'

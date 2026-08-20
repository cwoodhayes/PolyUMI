"""
Tests for the two chunk wire formats and the publisher that picks between them.

Three producers build these messages — the policy client and both on-arm probes — so the format
rules are tested once here rather than through each of them. What the individual nodes still test
is their own DEFAULT, since aiming a probe at the wrong executor is silent.
"""

import pytest
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectory

from polyumi_ros2.target_chunk import TargetChunkPublisher, Wire, multidof_trajectory


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    """Init/shutdown rclpy once for the whole module."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    """Build a bare node to hang publishers off."""
    n = Node('target_chunk_test')
    yield n
    n.destroy_node()


def _poses(n: int) -> list[Pose]:
    """`n` poses distinguishable by index, so ordering survives the round trip."""
    out = []
    for i in range(n):
        pose = Pose()
        pose.position.x = float(i)
        pose.orientation.w = 1.0
        out.append(pose)
    return out


def _stamp(seconds: float):
    """Build a builtin_interfaces Time, the way a caller's clock would hand one over."""
    return Time(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))


@pytest.mark.parametrize(
    ('wire', 'topic', 'msg_type'),
    [
        (Wire.MULTIDOF, '/polyumi/target_poses_traj', MultiDOFJointTrajectory),
        (Wire.POSE_ARRAY, '/polyumi/target_poses', PoseArray),
    ],
)
def test_each_wire_gets_its_own_topic_and_type(node, wire, topic, msg_type):
    """
    Separate topics per format, so both executors can run and neither sees a message it cannot parse.

    Publishing the wrong type at the right topic is silent — the other executor simply never
    subscribes, and nothing moves with no error anywhere.
    """
    pub = TargetChunkPublisher(node, wire=wire, frame_id='fr3_link0', joint_name='polyumi_tcp')

    assert pub.wire is wire
    assert pub.topic_name == topic
    assert pub._pub.msg_type is msg_type


def test_unknown_wire_fails_fast(node):
    """A typo must not leave a producer publishing where nothing subscribes."""
    with pytest.raises(ValueError):
        TargetChunkPublisher(node, wire='multidoff', frame_id='fr3_link0', joint_name='polyumi_tcp')


def test_multidof_times_use_the_preslice_index():
    """
    Waypoint k lands at ``stamp + (first_index + k) * dt``, NOT at ``stamp + k * dt``.

    Chunks are sliced by a stale-drop before publication. Numbering the survivors from zero would
    slide the whole timeline ``first_index * dt`` earlier, and the consumer reads these as absolute
    instants — so the arm would reach each pose that much too soon. A uniformly shifted chunk looks
    like tracking lag rather than a bug, which is why it is pinned.
    """
    msg = multidof_trajectory(
        _poses(5), frame_id='fr3_link0', joint_name='polyumi_tcp', stamp=_stamp(5.0), dt=0.1, first_index=3
    )

    offsets = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points]
    assert offsets == pytest.approx([0.3, 0.4, 0.5, 0.6, 0.7])
    assert [p.transforms[0].translation.x for p in msg.points] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_multidof_carries_the_frames_the_controller_matches_on():
    """The controller rejects chunks whose frame_id is not its base frame, so both must be set."""
    msg = multidof_trajectory(_poses(1), frame_id='fr3_link0', joint_name='polyumi_tcp', stamp=_stamp(5.0), dt=0.1)

    assert msg.header.frame_id == 'fr3_link0'
    assert msg.joint_names == ['polyumi_tcp']

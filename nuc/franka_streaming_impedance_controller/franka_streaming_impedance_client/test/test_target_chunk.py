"""
Tests for the chunk wire format and the publisher that builds it.

Every producer builds these messages through the same helpers, so the format rules are tested
once here rather than through each of them.
"""

import pytest
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectory

from franka_streaming_impedance_client.target_chunk import TargetChunkPublisher, multidof_trajectory


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


def test_publisher_uses_the_given_topic_and_message_type(node):
    """The publisher addresses the topic it was given and builds the right message type."""
    pub = TargetChunkPublisher(node, frame_id='fr3_link0', joint_name='fr3_hand_tcp', topic='/impedance/target_poses')

    assert pub.topic_name == '/impedance/target_poses'
    assert pub._pub.msg_type is MultiDOFJointTrajectory


def test_multidof_times_use_the_preslice_index():
    """
    Waypoint k lands at ``stamp + (first_index + k) * dt``, NOT at ``stamp + k * dt``.

    Chunks are sliced by a stale-drop before publication. Numbering the survivors from zero would
    slide the whole timeline ``first_index * dt`` earlier, and the consumer reads these as absolute
    instants — so the arm would reach each pose that much too soon. A uniformly shifted chunk looks
    like tracking lag rather than a bug, which is why it is pinned.
    """
    msg = multidof_trajectory(
        _poses(5), frame_id='fr3_link0', joint_name='fr3_hand_tcp', stamp=_stamp(5.0), dt=0.1, first_index=3
    )

    offsets = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points]
    assert offsets == pytest.approx([0.3, 0.4, 0.5, 0.6, 0.7])
    assert [p.transforms[0].translation.x for p in msg.points] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_multidof_carries_the_frames_the_controller_matches_on():
    """The controller rejects chunks whose frame_id is not its base frame, so both must be set."""
    msg = multidof_trajectory(_poses(1), frame_id='fr3_link0', joint_name='fr3_hand_tcp', stamp=_stamp(5.0), dt=0.1)

    assert msg.header.frame_id == 'fr3_link0'
    assert msg.joint_names == ['fr3_hand_tcp']

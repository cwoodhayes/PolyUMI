"""
The wire format for a chunk of target EEF poses, and one publisher that speaks it.

The streaming Cartesian impedance controller takes a ``MultiDOFJointTrajectory``, where every
waypoint has an absolute instant (``header.stamp + time_from_start``) that its interpolator
splices on. Every producer of target poses — the policy client and both on-arm probes — builds
this same message via ``TargetChunkPublisher``.
"""

from geometry_msgs.msg import Pose, PoseArray, Transform
from rclpy.duration import Duration
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

#: Where the streaming controller listens. Producers may override per-node, but this is the one
#: topic the running stack is wired for.
TARGET_POSES_TOPIC = '/polyumi/target_poses_traj'

#: What must be running for a chunk to reach the arm, for "nobody is listening" errors. The
#: controller can be loaded and still not subscribed, which is the confusing half.
CONSUMER_HINT = (
    'polyumi_cartesian_impedance_controller, ACTIVE — being loaded is not enough; '
    'check `ros2 control list_controllers` on the NUC'
)


def pose_array(poses: list[Pose], *, frame_id: str, stamp) -> PoseArray:
    """
    Build a PoseArray of `poses` in `frame_id`.

    `stamp` carries no per-waypoint meaning — the consumer re-times the chunk from arrival.
    """
    msg = PoseArray()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.poses = list(poses)
    return msg


def multidof_trajectory(
    poses: list[Pose],
    *,
    frame_id: str,
    joint_name: str,
    stamp,
    dt: float,
    first_index: int = 0,
) -> MultiDOFJointTrajectory:
    """
    Build a MultiDOFJointTrajectory placing waypoint k at ``stamp + (first_index + k) * dt``.

    `first_index` is the index of ``poses[0]`` in the UNSLICED chunk, because chunks are usually
    published with their leading waypoints dropped as stale. Numbering the survivors from zero
    would slide the whole timeline earlier, and the consumer reads these as absolute instants — a
    uniform shift that looks like tracking lag rather than a bug.
    """
    msg = MultiDOFJointTrajectory()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.joint_names = [joint_name]
    for i, pose in enumerate(poses):
        transform = Transform()
        transform.translation.x = pose.position.x
        transform.translation.y = pose.position.y
        transform.translation.z = pose.position.z
        transform.rotation = pose.orientation
        point = MultiDOFJointTrajectoryPoint()
        point.transforms = [transform]
        point.time_from_start = Duration(seconds=(first_index + i) * dt).to_msg()
        msg.points.append(point)
    return msg


class TargetChunkPublisher:
    """
    Publishes pose chunks as a timed MultiDOFJointTrajectory for the streaming controller.

    Wraps a plain publisher rather than replacing it, so callers keep ``topic_name`` and
    ``get_subscription_count()`` — publishing where nothing subscribes is otherwise silent, since
    nothing moves and there is no error anywhere.
    """

    def __init__(
        self,
        node,
        *,
        frame_id: str,
        joint_name: str,
        topic: str | None = None,
        qos: int = 10,
    ):
        """Create the underlying publisher, defaulting to :data:`TARGET_POSES_TOPIC`."""
        self._frame_id = frame_id
        self._joint_name = joint_name
        self._node = node
        self._pub = node.create_publisher(MultiDOFJointTrajectory, topic or TARGET_POSES_TOPIC, qos)

    @property
    def topic_name(self) -> str:
        """Resolved topic name, for log messages."""
        return self._pub.topic_name

    def get_subscription_count(self) -> int:
        """How many subscribers the topic has — i.e. whether the controller is listening."""
        return self._pub.get_subscription_count()

    def publish(self, poses: list[Pose], *, dt: float = 0.0, first_index: int = 0, stamp=None) -> None:
        """
        Publish `poses` as a MultiDOFJointTrajectory.

        `stamp` defaults to now; pass an earlier instant to command ahead of time, which is how
        action latency is compensated. See multidof_trajectory for what first_index indexes into.
        """
        stamp = stamp if stamp is not None else self._node.get_clock().now().to_msg()
        self._pub.publish(
            multidof_trajectory(
                poses,
                frame_id=self._frame_id,
                joint_name=self._joint_name,
                stamp=stamp,
                dt=dt,
                first_index=first_index,
            )
        )

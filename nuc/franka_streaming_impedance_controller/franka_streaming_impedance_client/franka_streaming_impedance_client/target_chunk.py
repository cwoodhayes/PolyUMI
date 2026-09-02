"""
How a chunk of target EEF poses is put on the wire — as a command, and as a picture of one.

``multidof_trajectory`` is the COMMAND format. The streaming Cartesian impedance controller takes a
``MultiDOFJointTrajectory``, where every waypoint has an absolute instant (``header.stamp +
time_from_start``) that its interpolator splices on. Every producer builds it through
``TargetChunkPublisher``, which is also what keeps ``topic_name`` and ``get_subscription_count()``
available for "nobody is listening" errors.

Two fields the controller matches on, and rejects the chunk over:

- ``header.frame_id`` must be the controller's ``base_frame``.
- ``joint_names`` must be exactly ``[tcp_frame]`` — it names the commanded frame, not a joint.

``pose_array`` is the PREVIEW format, and moves nothing. A producer can publish every commanded
chunk as an untimed ``PoseArray`` on a separate topic so the motion can be watched in rviz or
Foxglove whether or not execution is enabled. It lives here so both views of a chunk are built
from one place.
"""

from geometry_msgs.msg import Pose, PoseArray, Transform
from rclpy.duration import Duration
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint


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
        topic: str,
        qos: int = 10,
    ):
        """
        Create the underlying publisher.

        `topic` is explicit and has no default: the controller's own default is node-relative
        (``~/target_poses``, i.e. under the controller's name), which would resolve against the
        PUBLISHING node here and silently address nothing.
        """
        self._frame_id = frame_id
        self._joint_name = joint_name
        self._node = node
        self._pub = node.create_publisher(MultiDOFJointTrajectory, topic, qos)

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

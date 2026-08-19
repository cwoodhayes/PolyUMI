"""
Wire formats for a chunk of target EEF poses, and one publisher that speaks either.

Two executors consume target poses and they disagree about time:

* ``fr3_moveit_bridge`` takes a ``PoseArray``. It carries no timing at all — the bridge re-times
  the chunk from whenever it happens to arrive.
* the streaming Cartesian impedance controller takes a ``MultiDOFJointTrajectory``, where every
  waypoint has an absolute instant (``header.stamp + time_from_start``) that its interpolator
  splices on.

Every producer of target poses — the policy client and both on-arm probes — therefore has to build
one of two messages from the same list of poses, and has to be switchable between them while both
executors exist. That is this module. It goes away with the ``PoseArray`` half once the MoveIt path
is retired, at which point ``TargetChunkPublisher`` collapses to the trajectory builder.
"""

from enum import StrEnum

from geometry_msgs.msg import Pose, PoseArray, Transform
from rclpy.duration import Duration
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint


class Wire(StrEnum):
    """
    Which executor a chunk is aimed at, and therefore how it is encoded.

    A StrEnum so it drops straight into a ROS string parameter and compares equal to the literal,
    while ``Wire(value)`` does the validation — a typo raises rather than silently publishing to a
    topic nobody is subscribed to.
    """

    #: Untimed poses, consumed by fr3_moveit_bridge.
    POSE_ARRAY = 'pose_array'
    #: Absolutely-timed poses, consumed by the streaming impedance controller.
    MULTIDOF = 'multidof'

    @property
    def default_topic(self) -> str:
        """One topic per format, so neither executor is ever handed a message it cannot parse."""
        return {
            Wire.POSE_ARRAY: '/polyumi/target_poses',
            Wire.MULTIDOF: '/polyumi/target_poses_traj',
        }[self]

    @property
    def consumer(self) -> str:
        """
        What must be running for this format to reach the arm, for 'nobody is listening' errors.

        The two fail differently: the bridge is a process that is either up or not, while the
        controller can be loaded and still not subscribed because it was never activated.
        """
        return {
            Wire.POSE_ARRAY: ('fr3_moveit_bridge (ros2 launch nuc/launch/fr3_inference.launch.py on the NUC)'),
            Wire.MULTIDOF: (
                'polyumi_cartesian_impedance_controller, ACTIVE — being loaded is not enough; '
                'check `ros2 control list_controllers` on the NUC'
            ),
        }[self]


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
    Publishes pose chunks in whichever wire format the running executor speaks.

    Wraps a plain publisher rather than replacing it, so callers keep ``topic_name`` and
    ``get_subscription_count()`` — aiming at the wrong executor is otherwise silent, since the
    other one simply never subscribes and nothing moves with no error anywhere.
    """

    def __init__(
        self,
        node,
        *,
        wire: str,
        frame_id: str,
        joint_name: str,
        topic: str | None = None,
        qos: int = 10,
    ):
        """
        Create the underlying publisher for `wire`, defaulting the topic to its default_topic.

        :raises ValueError: if `wire` is not a known format.
        """
        self._wire = Wire(wire)
        self._frame_id = frame_id
        self._joint_name = joint_name
        self._node = node
        msg_type = PoseArray if self._wire is Wire.POSE_ARRAY else MultiDOFJointTrajectory
        self._pub = node.create_publisher(msg_type, topic or self._wire.default_topic, qos)

    @property
    def wire(self) -> Wire:
        """Which wire format this publisher emits."""
        return self._wire

    @property
    def topic_name(self) -> str:
        """Resolved topic name, for log messages."""
        return self._pub.topic_name

    def get_subscription_count(self) -> int:
        """How many subscribers the topic has — i.e. whether an executor is listening."""
        return self._pub.get_subscription_count()

    def publish(self, poses: list[Pose], *, dt: float = 0.0, first_index: int = 0, stamp=None) -> None:
        """
        Publish `poses` in this publisher's wire format.

        `dt` and `first_index` are ignored for PoseArray, which carries no timing. `stamp` defaults
        to now; pass an earlier instant to command ahead of time, which is how action latency is
        compensated. See multidof_trajectory for what first_index indexes into.
        """
        stamp = stamp if stamp is not None else self._node.get_clock().now().to_msg()
        if self._wire is Wire.POSE_ARRAY:
            self._pub.publish(pose_array(poses, frame_id=self._frame_id, stamp=stamp))
        else:
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

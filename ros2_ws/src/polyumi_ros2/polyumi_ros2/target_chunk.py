"""
PolyUMI's wiring for the streaming impedance controller's chunk format.

The format itself — how a chunk of target EEF poses is put on the wire — lives in
``franka_streaming_impedance_client.target_chunk``, which is a standalone open-source package
shared with other users of the controller. What is PolyUMI-specific, and so stays here, is
*which* topic this deployment publishes on and *what* has to be running to receive it.

Import ``TargetChunkPublisher`` from here rather than from the generic package: the subclass
below is the one that knows this deployment's topic. ``pose_array`` is re-exported unchanged —
it is the PREVIEW format ``policy_client_node`` publishes so a chunk can be watched in Foxglove
whether or not execution is enabled.
"""

from franka_streaming_impedance_client.target_chunk import TargetChunkPublisher as _ChunkPublisher
from franka_streaming_impedance_client.target_chunk import pose_array

__all__ = ['CONSUMER_HINT', 'TARGET_POSES_TOPIC', 'TargetChunkPublisher', 'pose_array']

#: Where the streaming controller listens. Producers may override per-node, but this is the one
#: topic the running stack is wired for. Set as `target_topic` in nuc/config/polyumi_controllers.yaml.
TARGET_POSES_TOPIC = '/polyumi/target_poses_traj'

#: What must be running for a chunk to reach the arm, for "nobody is listening" errors. The
#: controller can be loaded and still not subscribed, which is the confusing half.
CONSUMER_HINT = (
    'polyumi_cartesian_impedance_controller, ACTIVE — being loaded is not enough; '
    'check `ros2 control list_controllers` on the NUC'
)


class TargetChunkPublisher(_ChunkPublisher):
    """
    The generic chunk publisher, defaulted to the topic this deployment is wired for.

    The generic class requires `topic`, because its own node-relative default would resolve
    against the publishing node and address nothing. PolyUMI has one answer for every producer,
    so it is supplied here instead of at each of the four call sites.
    """

    def __init__(self, node, *, frame_id: str, joint_name: str, topic: str | None = None, qos: int = 10):
        """Create the publisher, defaulting to :data:`TARGET_POSES_TOPIC`."""
        super().__init__(
            node,
            frame_id=frame_id,
            joint_name=joint_name,
            topic=topic or TARGET_POSES_TOPIC,
            qos=qos,
        )

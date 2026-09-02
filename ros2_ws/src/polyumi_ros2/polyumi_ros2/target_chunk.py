"""
PolyUMI's wiring for the streaming impedance controller's chunk format.

The format itself — how a chunk of target EEF poses is put on the wire — lives in
``franka_streaming_impedance_client.target_chunk``, which is a standalone open-source package
shared with other users of the controller. What is PolyUMI-specific, and so stays here, is
*which* topic this deployment publishes on and *what* has to be running to receive it.

Re-exported so producers keep one import: ``TargetChunkPublisher`` for the COMMAND format
(absolutely-timed ``MultiDOFJointTrajectory``, spliced by the controller's interpolator), and
``pose_array`` for the PREVIEW format that ``policy_client_node`` publishes so a chunk can be
watched in Foxglove whether or not execution is enabled.
"""

from franka_streaming_impedance_client.target_chunk import (
    TargetChunkPublisher,
    multidof_trajectory,
    pose_array,
)

__all__ = [
    'CONSUMER_HINT',
    'TARGET_POSES_TOPIC',
    'TargetChunkPublisher',
    'multidof_trajectory',
    'pose_array',
]

#: Where the streaming controller listens. Producers may override per-node, but this is the one
#: topic the running stack is wired for. Set as `target_topic` in nuc/config/polyumi_controllers.yaml.
TARGET_POSES_TOPIC = '/polyumi/target_poses_traj'

#: What must be running for a chunk to reach the arm, for "nobody is listening" errors. The
#: controller can be loaded and still not subscribed, which is the confusing half.
CONSUMER_HINT = (
    'polyumi_cartesian_impedance_controller, ACTIVE — being loaded is not enough; '
    'check `ros2 control list_controllers` on the NUC'
)

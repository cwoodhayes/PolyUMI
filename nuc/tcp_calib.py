"""
The one definition of PolyUMI's TCP frame on the FR3 — where the *policy* thinks the EEF is.

The policy is trained on poses expressed at the closed-fingertip midpoint, in GoPro **optical**
axes (x right, y down, z forward). See ``T_gopro_to_fingertip`` in ingest/config/gripper_calib.yaml
and preproc step 5 (``eef_pose_step.py``), which retargets every exported pose onto that frame.
The stock ``fr3_hand_tcp`` is a different physical point in a different axis convention, so
observing or commanding it feeds the policy a frame it was never trained on.

``polyumi_tcp`` is that fingertip frame, as a fixed child of ``fr3_hand``. Two consumers need it
and they MUST agree, which is why the numbers live here and nowhere else:

  * TF, for the laptop's observation lookup (``policy_client_node``'s ``eef_frame``) and for
    seeing the frame in Foxglove — published by a static_transform_publisher in
    fr3_bringup.launch.py.
  * move_group's RobotModel, for ``GetCartesianPath.link_name`` (``fr3_moveit_bridge``) — passed
    as xacro args into nuc/description/fr3_polyumi.urdf.xacro by fr3_move_group.launch.py.

Where the numbers come from
---------------------------
The rotation is exact, from geometry alone — no measurement, no ambiguity beyond one sign:

  * ``fr3_hand`` has z along the approach axis and moves its fingers along ±y
    (``franka_hand.xacro``: ``finger_joint1 axis="0 1 0"``).
  * the policy frame is optical, so its z is the camera's forward axis — which is the approach
    axis (``T_gripper_base_to_gopro`` maps gopro z to gripper-forward) — and its x is the
    finger-opening axis, since the tagged fingers open left-right in the camera image.

Sharing z and mapping x onto ±y leaves exactly Rz(±90°). The sign is which finger lands on
camera-left; flip ``TCP_RPY``'s yaw if the frame comes out mirrored in Foxglove (verification
step 4 in the plan catches this by eye, before anything moves).

The translation is a CAD read: the fingertip-midpoint origin — where the closed tips meet, on the
plane of the fingers' upper surface, coplanar with the ArUco tags — expressed in ``fr3_hand``.
This is the same method UMI uses; they hardcode their camera→TCP transform from CAD too
(``06_generate_dataset_plan.py``, ``franka_interpolation_controller.py``) and ship no calibration
procedure for it at all.

**``TCP_XYZ`` is deliberately not the true fingertip right now.** It carries
``LEGACY_TRAINING_Y_ERROR`` on top, so that it names the frame the checkpoint in use was actually
trained in. What matters for a policy is that training and inference agree on the body frame, not
that the frame is anatomically correct — see that constant for the full story and for how to
retire it.

It cross-checks against the handheld gripper's chain to ~2 mm. ``T_gopro_to_fingertip`` puts the
fingertips 0.259 m forward of the GoPro; this file puts them 0.2569 m forward of ``fr3_hand``. The
two agreeing means the camera's sensor plane sits within a couple of mm of the ``fr3_hand`` plane —
consistent with the mount, and the sort of thing that would be off by centimetres if either
measurement were wrong.

**Verified on hardware, 2026-08-07** (``ros2 run polyumi_ros2 tcp_pivot_test``): with
``LEGACY_TRAINING_Y_ERROR`` zeroed, pivoting about the TCP holds the closed fingertips visibly
still. So the geometry below — including the ``Rz(+90°)`` sign, which a pivot test would expose as
a ~15 cm sweep if mirrored — is right. Note this validates the FINGERTIP frame, not ``TCP_XYZ`` as
shipped, which deliberately sits off it while the legacy offset is non-zero.

Residual error, deliberately not chased: real GoPro mount tilt versus the CAD-nominal "optical
axis parallel to the approach axis". That is what a camera-based hand-eye calibration would buy.
Suspect it first if the policy is systematically off in *orientation* rather than position.
"""

import math

TCP_PARENT = 'fr3_hand'
TCP_CHILD = 'polyumi_tcp'

# Along the approach axis, in two independently sourced halves so each stays checkable:
# franka_description puts the finger carriage plane here (both finger_joint origins are
# xyz="0 0 0.0584" in fr3_hand), and the PolyUMI fingers are measured from that plane out.
FINGER_CARRIAGE_Z = 0.0584   # fr3_hand -> the plane the fingers translate in
CARRIAGE_TO_FINGERTIP_Z = 0.1985  # that plane -> the closed fingertips, PolyUMI CAD

# Across the fingers: from the fr3_hand origin up to the fingers' upper surface, the plane the
# ArUco tags sit on. `up` is +x because x is the hand's thin axis; y is the finger-travel axis.
# fr3_hand sits on the fingers' x-midline — not an assumption, franka_description's own geometry
# is symmetric about x=0 (hand mesh spans -0.0316..+0.0316, stock finger -0.0105..+0.0105).
FINGERTIP_X = 0.019612

# ponytail: a temporary train/inference pairing, not geometry. Set to 0.0 once a checkpoint
# trained on corrected data is in use — that is the whole upgrade path.
#
# T_gopro_to_fingertip's y was 0.014 until 2026-08-06 and is now 0.07069 (the old value put the
# GoPro sensor roughly flush with the top of the hand shell, which the mount plainly isn't).
# Checkpoints exported before that fix therefore learned a body frame sitting this far toward the
# camera from the real fingertips — a perfectly self-consistent frame, just not the fingertips.
# Inference has to speak the frame the policy was trained in, so we shift the TCP to match rather
# than re-export and retrain. Optical +y is "down", so a too-small y means the trained frame is
# ABOVE the fingertips, i.e. further along +x of fr3_hand.
LEGACY_TRAINING_Y_ERROR = 0.07069 - 0.014

# y is 0 by symmetry: the fingers close on the y=0 plane, so the closed-fingertip midpoint is on it.
TCP_XYZ = (
    FINGERTIP_X + LEGACY_TRAINING_Y_ERROR,
    0.0,
    FINGER_CARRIAGE_Z + CARRIAGE_TO_FINGERTIP_Z,
)

# Rz(+90°), not -90°: the policy's y is the camera's "down", which points from the tagged upper
# surface into the finger body, i.e. -x of fr3_hand. With z shared and x_policy = +y_hand, that
# fixes the sign. Confirm it by eye in Foxglove anyway (docs/crb-fr3-inference.md).
TCP_RPY = (0.0, 0.0, math.pi / 2)


def xacro_args() -> list[str]:
    """Build the ``name:=value`` pairs fr3_polyumi.urdf.xacro requires, for a launch Command."""
    return [
        f' polyumi_tcp_xyz:="{" ".join(str(v) for v in TCP_XYZ)}"',
        f' polyumi_tcp_rpy:="{" ".join(str(v) for v in TCP_RPY)}"',
    ]


def static_transform_publisher_args() -> list[str]:
    """Build the tf2_ros static_transform_publisher argv publishing TCP_PARENT -> TCP_CHILD."""
    x, y, z = TCP_XYZ
    roll, pitch, yaw = TCP_RPY
    return [
        '--x', str(x), '--y', str(y), '--z', str(z),
        '--roll', str(roll), '--pitch', str(pitch), '--yaw', str(yaw),
        '--frame-id', TCP_PARENT, '--child-frame-id', TCP_CHILD,
    ]


def describe() -> str:
    """One-line summary of the TCP in force, logged at bringup so it is never a mystery."""
    return f'{TCP_PARENT} -> {TCP_CHILD}: xyz={TCP_XYZ} rpy={TCP_RPY} (nuc/tcp_calib.py)'

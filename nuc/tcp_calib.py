"""
The measured constants of what PolyUMI bolts to the FR3's flange: the TCP frame, and the payload.

The TCP frame is where the *policy* thinks the EEF is; the payload is what that assembly weighs.

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

It cross-checks against the handheld gripper's chain to ~2 mm. ``T_gopro_to_fingertip`` puts the
fingertips 0.259 m forward of the GoPro; this file puts them 0.2569 m forward of ``fr3_hand``. The
two agreeing means the camera's sensor plane sits within a couple of mm of the ``fr3_hand`` plane —
consistent with the mount, and the sort of thing that would be off by centimetres if either
measurement were wrong.

**Verified on hardware, 2026-08-07** (``ros2 run polyumi_ros2 tcp_pivot_test``): pivoting about
this TCP holds the closed fingertips visibly still, so the geometry below — including the
``Rz(+90°)`` sign, which a mirrored version would expose as a ~15 cm sweep — is right.

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
FINGER_CARRIAGE_Z = 0.0584  # fr3_hand -> the plane the fingers translate in
CARRIAGE_TO_FINGERTIP_Z = 0.1985  # that plane -> the closed fingertips, PolyUMI CAD

# Across the fingers: from the fr3_hand origin up to the fingers' upper surface, the plane the
# ArUco tags sit on. `up` is +x because x is the hand's thin axis; y is the finger-travel axis.
# fr3_hand sits on the fingers' x-midline — not an assumption, franka_description's own geometry
# is symmetric about x=0 (hand mesh spans -0.0316..+0.0316, stock finger -0.0105..+0.0105).
FINGERTIP_X = 0.019612

# y is 0 by symmetry: the fingers close on the y=0 plane, so the closed-fingertip midpoint is on it.
TCP_XYZ = (FINGERTIP_X, 0.0, FINGER_CARRIAGE_Z + CARRIAGE_TO_FINGERTIP_Z)

# Rz(+90°), not -90°: the policy's y is the camera's "down", which points from the tagged upper
# surface into the finger body, i.e. -x of fr3_hand. With z shared and x_policy = +y_hand, that
# fixes the sign. Confirm it by eye in Foxglove anyway (docs/crb-fr3-inference.md).
TCP_RPY = (0.0, 0.0, math.pi / 2)

# --------------------------------------------------------------------------------------------
# The hand subtree, which robot_state_publisher normally publishes from the URDF.
#
# franka.launch.py feeds its one `load_gripper` flag to both `xacro hand:=` and the franka_gripper
# include. PolyUMI's own hand driver owns the libfranka connection, so franka_gripper must not run
# — which takes the whole hand out of robot_description as a side effect. fr3_bringup.launch.py
# republishes the FIXED joints statically to fill that hole.
#
# Two of them, and both have a consumer that fails hard without it:
#
#   fr3_link8 -> fr3_hand      polyumi_tcp hangs off fr3_hand, so losing this orphans it and the
#                              laptop's base -> polyumi_tcp lookup fails — no observation at all.
#   fr3_hand  -> fr3_hand_tcp  franka's O_T_EE frame. polyumi_cartesian_impedance_controller looks
#                              up fr3_hand_tcp -> polyumi_tcp once at activation to convert the
#                              reported EE pose onto the policy's TCP, and refuses to activate
#                              without it rather than silently controlling the wrist.
#
# Values are franka_description's own joint origins, read out of the rendered URDF (`xacro
# fr3.urdf.xacro hand:=true`) and matching the defaults in
# end_effectors/franka_hand/franka_hand_arguments.xacro. Geometry, not measurements — the -45 deg
# yaw is the flange's standard mounting rotation.
#
# The two FINGER joints are deliberately absent: fr3_finger_joint1/2 are PRISMATIC, so a static
# transform would assert an aperture the fingers do not have. That is why there is still no finger
# TF under load_gripper:=false — see docs/crb-fr3-inference.md.
HAND_STATIC_TRANSFORMS = (
    ('fr3_link8', 'fr3_hand', (0.0, 0.0, 0.0), (0.0, 0.0, -math.pi / 4)),
    ('fr3_hand', 'fr3_hand_tcp', (0.0, 0.0, 0.1034), (0.0, 0.0, 0.0)),
)


# --------------------------------------------------------------------------------------------
# The payload — everything past the flange, as libfranka's gravity compensation needs to know it.
#
# The FCI cancels gravity for the mass it knows about (m_ee + m_load) and nothing else. Whatever is
# unmodelled is a constant force the cartesian impedance spring has to fight, and it settles at
# dz = m * g / K_trans — a steady-state droop the moment the controller activates. Pushed once at
# bringup via franka_msgs/srv/SetLoad; see fr3_bringup.launch.py.
#
# MASS is the *residual*, not the whole assembly: Desk's end-effector config already supplies m_ee
# (the Franka Hand, 0.73 kg), so this covers only the GoPro, its mount and the PolyUMI fingers. If
# Desk's end effector is set to "None" it has to carry the hand too. Check which before touching
# the number:
#
#     ros2 topic echo /franka_robot_state_broadcaster/robot_state --field inertia_ee --once
#
# Note franka_hardware reads no payload from the URDF (only robot_ip and arm_id), so an <inertial>
# block in fr3_polyumi.urdf.xacro would change nothing here — setLoad or Desk are the only levers.
PAYLOAD_MASS = 0.5  # kg

# In fr3_hand, which uses the same frame orientation as TCP_XYZ, so the two are directly comparable
# (see the linear offset in TCP_XYZ above, positioned at the fingertips). setLoad wants it in the FLANGE,
# which is a 45 deg yaw away — payload_com_flange() does
# that conversion off HAND_STATIC_TRANSFORMS. Do not write flange coordinates here.

# guesswork; I'd say CoM is about 1/3 of the way along the fingers due to GoPro being far back.
# and then about 40mm up (which is neg. in this frame) from the finger surface, again due to the gopro
# and the x is a bit to the right, looking out from the camera, due to the shape of the gopro.
PAYLOAD_COM_HAND = (0.01, TCP_XYZ[1] - 0.04, TCP_XYZ[2] / 3)  # m

# The FR3 firmware validates the tensor before it looks at anything else, and a nonzero mass with a
# zero inertia is a physical impossibility it rejects outright:
#
#     libfranka: Set Load command rejected: invalid argument!
#
# So this cannot be left at zero the way UMI, SERL and polymetis leave it — they all reach the load
# through Desk, which derives a tensor for them. It is approximated as a uniform solid box, which is
# both good enough (inertia only enters the acceleration terms, and we move slowly) and self-
# consistent by construction: the principal moments of a real box always satisfy the triangle
# inequality the firmware checks, at whatever mass.
#
# Bounding box of the whole assembly past the flange, metres, ordered (x, y, z) in fr3_hand
# fr3_hand frame is at the flange. z is approach axis out through the fingers, x is up (ie out from
# the ArUco tag face of the fingers), and y is to the right (along the finger travel axis)
# Can estimate this from the CAD. These numbers generate a symmetrical bounding box.
# it is then shifted to be centered at the center of mass above.
PAYLOAD_EXTENTS = (0.12, 0.16, 0.27)


def _stp_args(xyz, rpy, parent, child) -> list[str]:
    """tf2_ros static_transform_publisher argv for one fixed transform."""
    x, y, z = xyz
    roll, pitch, yaw = rpy
    return [
        '--x', str(x), '--y', str(y), '--z', str(z),
        '--roll', str(roll), '--pitch', str(pitch), '--yaw', str(yaw),
        '--frame-id', parent, '--child-frame-id', child,
    ]  # fmt: skip


def xacro_args() -> list[str]:
    """Build the ``name:=value`` pairs fr3_polyumi.urdf.xacro requires, for a launch Command."""
    return [
        f' polyumi_tcp_xyz:="{" ".join(str(v) for v in TCP_XYZ)}"',
        f' polyumi_tcp_rpy:="{" ".join(str(v) for v in TCP_RPY)}"',
    ]


def static_transform_publisher_args() -> list[str]:
    """Build the tf2_ros static_transform_publisher argv publishing TCP_PARENT -> TCP_CHILD."""
    return _stp_args(TCP_XYZ, TCP_RPY, TCP_PARENT, TCP_CHILD)


def hand_transform_publishers() -> list[tuple[str, list[str]]]:
    """
    Build ``(node name, argv)`` for every fixed hand joint the URDF loses with load_gripper:=false.

    :returns: one entry per transform in :data:`HAND_STATIC_TRANSFORMS`.
    """
    return [
        (f'{child}_static_tf', _stp_args(xyz, rpy, parent, child)) for parent, child, xyz, rpy in HAND_STATIC_TRANSFORMS
    ]


def payload_com_flange() -> tuple[float, float, float]:
    """
    Convert :data:`PAYLOAD_COM_HAND` into ``fr3_link8``, which is the frame setLoad's F_x_Cload is in.

    The rotation is read out of :data:`HAND_STATIC_TRANSFORMS` rather than re-typed, so the flange's
    mounting yaw stays defined once. That entry's translation is (0, 0, 0) — the two origins
    coincide — so this is a pure rotation.
    """
    (_, _, _, (_, _, yaw)) = next(t for t in HAND_STATIC_TRANSFORMS if t[1] == TCP_PARENT)
    x, y, z = PAYLOAD_COM_HAND
    return (
        x * math.cos(yaw) - y * math.sin(yaw),
        x * math.sin(yaw) + y * math.cos(yaw),
        z,
    )


def payload_inertia() -> tuple[float, ...]:
    """
    Approximate the payload's inertia tensor as a uniform solid box, column-major about its CoM.

    Computed rather than written down so it stays consistent with :data:`PAYLOAD_MASS`: the FR3
    rejects a nonzero mass carrying a zero tensor, and a stale hand-typed tensor is the easy way
    back into that. Off-diagonal terms are zero — the box is taken as axis-aligned.
    """
    w, d, h = PAYLOAD_EXTENTS
    k = PAYLOAD_MASS / 12.0
    ixx, iyy, izz = k * (d**2 + h**2), k * (w**2 + h**2), k * (w**2 + d**2)
    return (ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)


def set_load_request() -> str:
    """Build the franka_msgs/srv/SetLoad request literal for a `ros2 service call`."""
    com = ', '.join(str(v) for v in payload_com_flange())
    inertia = ', '.join(str(v) for v in payload_inertia())
    return f'{{mass: {PAYLOAD_MASS}, center_of_mass: [{com}], load_inertia: [{inertia}]}}'


def describe() -> str:
    """One-line summary of the TCP in force, logged at bringup so it is never a mystery."""
    return f'{TCP_PARENT} -> {TCP_CHILD}: xyz={TCP_XYZ} rpy={TCP_RPY} (nuc/tcp_calib.py)'


def describe_payload() -> str:
    """One-line summary of the payload pushed to the FCI, for the same reason as :func:`describe`."""
    com = tuple(round(v, 4) for v in payload_com_flange())
    return f'payload: {PAYLOAD_MASS} kg, CoM {com} in fr3_link8 (nuc/tcp_calib.py)'

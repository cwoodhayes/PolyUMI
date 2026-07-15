"""Shared rigid-transform helpers for optitrack pose conversion."""

import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation


def transform_optitrack_pose(o_pose: np.ndarray, T_gb_rb: RigidTransform, T_gb_gp: RigidTransform) -> np.ndarray:
    """
    Transform an OptiTrack rigid-body pose to the GoPro frame in optitrack coordinates.

    Args:
        o_pose: OptiTrack rigid-body pose in optitrack frame (T_o_rb). (7,) [x y z qx qy qz qw]
        T_gb_rb: Pose of the optitrack rigid body in the gripper-base frame.
        T_gb_gp: Pose of the GoPro frame in the gripper-base frame.

    Returns:
        GoPro pose in optitrack frame. (7,) [x y z qx qy qz qw]

    """
    T_o_rb = RigidTransform.from_components(
        translation=o_pose[:3],
        rotation=Rotation.from_quat(o_pose[3:]),
    )
    T_o_gp = T_o_rb * T_gb_rb.inv() * T_gb_gp
    out = np.zeros(7)
    out[:3] = T_o_gp.translation
    out[3:] = T_o_gp.rotation.as_quat()
    return out


def retarget_body_frame(poses: np.ndarray, T_x_target: RigidTransform) -> np.ndarray:
    """
    Re-express a pose trajectory measured at body frame X onto another body frame.

    Computes ``T_w_target = T_w_x · T_x_target``.

    Only the *body* frame changes; the world frame w is left as-is. That asymmetry is the
    point: a shared world frame cancels out of the relative pose representation the policy
    trains on (inv(T_0)·T_k is invariant to a global re-frame), but a body-frame offset does
    not — it conjugates the relative transform, so a rotation of R about a body offset x
    leaks a (R - I)·x error into the relative translation.

    Args:
        poses: (N, 7) [x y z qx qy qz qw] — T_w_x, pose of X in some world frame w.
        T_x_target: pose of the target frame expressed in X.

    Returns:
        (N, 7) T_w_target — the target frame's pose in the same world frame w. Rows whose
        input was NaN (e.g. SLAM tracking loss) stay NaN rather than raising.

    """
    poses = np.asarray(poses, dtype=np.float64)
    out = np.full((len(poses), 7), np.nan, dtype=np.float64)
    valid = ~np.isnan(poses).any(axis=1)
    if not valid.any():
        return out
    T_w_x = RigidTransform.from_components(
        translation=poses[valid, :3],
        rotation=Rotation.from_quat(poses[valid, 3:]),
    )
    T_w_target = T_w_x * T_x_target
    out[valid, :3] = T_w_target.translation
    out[valid, 3:] = T_w_target.rotation.as_quat()
    return out


def gopro_to_hand_transform(calib: dict) -> RigidTransform:
    """
    Build T_gp_hand — the pose of the hand frame expressed in the GoPro frame.

    This is the one transform that must hold on *both* embodiments. The handheld gripper's
    ``gripper_base`` is a mechanical part the Franka end-effector does not have, so a frame
    defined against it cannot be reconstructed on the robot. The GoPro-to-fingers geometry is
    shared by construction, so a frame defined against the GoPro can be — which is what makes
    the trajectories comparable across demo and deployment.

    Raises:
        KeyError: if the calibration has no ``T_gopro_to_hand`` entry. Deliberately not
            defaulted: silently falling back to some other frame is precisely the failure
            this chain exists to prevent.

    """
    try:
        hand = calib['T_gopro_to_hand']
    except KeyError:
        raise KeyError(
            'gripper_calib.yaml has no T_gopro_to_hand entry. It defines the hand frame '
            'relative to the GoPro and is required to export poses the robot can reproduce; '
            'there is no safe default.'
        )
    return RigidTransform.from_components(
        translation=np.array(hand['translation'], dtype=float),
        rotation=Rotation.from_quat(hand['rotation']),
    )


def gripper_calib_transforms(calib: dict) -> tuple[RigidTransform, RigidTransform, RigidTransform]:
    """
    Build (T_gb_rb, T_gb_gp, T_o_w) RigidTransforms from a gripper_calib zarr-attrs dict.

    Returns:
        (T_gb_rb, T_gb_gp, T_o_w) as RigidTransform objects.

    """
    rb = calib['T_gripper_base_to_optitrack_rigid_body']
    gp = calib['T_gripper_base_to_gopro']
    world = calib['T_optitrack_to_world']
    T_gb_rb = RigidTransform.from_components(
        translation=np.array(rb['translation'], dtype=float),
        rotation=Rotation.from_quat(rb['rotation']),
    )
    T_gb_gp = RigidTransform.from_components(
        translation=np.array(gp['translation'], dtype=float),
        rotation=Rotation.from_quat(gp['rotation']),
    )
    T_o_w = RigidTransform.from_components(
        translation=np.array(world['translation'], dtype=float),
        rotation=Rotation.from_quat(world['rotation']),
    )
    return T_gb_rb, T_gb_gp, T_o_w

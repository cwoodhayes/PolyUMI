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

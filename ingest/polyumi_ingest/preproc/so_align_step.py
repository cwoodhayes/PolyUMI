"""SLAM-to-optitrack alignment preprocessing step."""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from polyumi_ingest.config import load_gripper_calib
from polyumi_ingest.preproc.step_base import PreprocessingStep, register_preprocessing_step
from polyumi_ingest.pzarr.store import arr, grp
from polyumi_ingest.transforms import gripper_calib_transforms, transform_optitrack_pose

log = logging.getLogger(__name__)


def _calc_rms_errors(
    R: np.ndarray,
    t: np.ndarray,
    slam_pos: np.ndarray,
    slam_quat: np.ndarray,
    ot_pos: np.ndarray,
    ot_quat: np.ndarray,
) -> tuple[float, float]:
    """
    Compute RMS position and orientation errors between aligned SLAM and OptiTrack poses.

    Args:
        R: (3, 3) rotation from SVD alignment.
        t: (3,) translation from SVD alignment.
        slam_pos: (N, 3) SLAM positions.
        slam_quat: (N, 4) SLAM quaternions (xyzw).
        ot_pos: (N, 3) OptiTrack positions.
        ot_quat: (N, 4) OptiTrack quaternions (xyzw), unnormalised.

    Returns:
        (rms_pos_m, rms_rot_deg): position RMS in metres, orientation RMS in degrees.

    """
    residuals = (R @ slam_pos.T).T + t - ot_pos
    rms_pos = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))

    # Rotate SLAM orientations into the OptiTrack frame before comparing.
    R_rot = Rotation.from_matrix(R)
    slam_rots = R_rot * Rotation.from_quat(slam_quat)
    ot_rots = Rotation.from_quat(ot_quat / np.linalg.norm(ot_quat, axis=1, keepdims=True))
    angle_errors_deg = (slam_rots.inv() * ot_rots).magnitude() * (180.0 / np.pi)
    rms_rot = float(np.sqrt(np.mean(angle_errors_deg**2)))

    return rms_pos, rms_rot


def _svd_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for R, t minimising sum ||R @ src_i + t - dst_i||^2 (Horn's method).

    Args:
        src: (N, 3) positions in source frame.
        dst: (N, 3) corresponding positions in destination frame.

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation.  dst_p =(approx) R @ src_p + t.

    See https://roboticsknowledgebase.com/wiki/math/registration-techniques/
    for a good explanation of this algorithm.

    """
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    H = (src - c_src).T @ (dst - c_dst)
    U, _, Vt = np.linalg.svd(H)
    # Correct for reflections: ensure det(R) = +1.
    D = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T
    t = c_dst - R @ c_src
    return R, t


@register_preprocessing_step(step_number=3, step_name='slam-optitrack-align')
class SlamToWorldAlignStep(PreprocessingStep):
    """
    Compute and store the optitrack-to-slam rigid transform T_ws.

    Only runs if there is both slam and optitrack data to align; otherwise stores an identity transform.
    For each episode that has SLAM poses, collects (timestamp, slam_position) pairs
    and applies any time-sync correction so the timestamps align with the OptiTrack
    clock domain.  OptiTrack poses are converted to optitrack-frame GoPro positions via
    the gripper calibration chain.  Within the time overlap window, OptiTrack
    positions are interpolated to each SLAM timestamp, producing matched point pairs.
    Horn's closed-form SVD method then solves for the rotation R and translation t
    that minimise::

        Σ ‖R · p_slam_i + t − p_optitrack_i‖²

    The result is written to the root zarr attrs as ``optitrack_to_slam_transform``
    with keys ``translation`` (list of 3 floats) and ``rotation`` (list of 4
    floats, xyzw).  The MCAP exporter reads this attr and uses it as the static
    ``optitrack → slam`` frame transform.

    Prerequisites: OptiTrack data in the root group (``optitrack/pose`` and
    ``optitrack/timestamps``) and at least one episode with ``gopro/slam_poses``.
    """

    @staticmethod
    def _write_identity(root: zarr.Group) -> None:
        """Store an identity transform as optitrack_to_slam_transform."""
        root.attrs['optitrack_to_slam_transform'] = {
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],
        }

    def run_step(self, scene_zarr: pathlib.Path, force: bool = False) -> None:
        """Compute T_ws and write it to the root zarr attrs."""
        root = zarr.open_group(str(scene_zarr), mode='a')

        if 'optitrack_to_slam_transform' in root.attrs and not force:
            log.info('optitrack_to_slam_transform already present; use --force to recompute.')
            return

        if 'optitrack/pose' not in root:
            log.warning('No optitrack data in scene; storing identity T_ws.')
            self._write_identity(root)
            return

        gripper_calib = load_gripper_calib()
        root.attrs['gripper_calib'] = gripper_calib

        T_gb_rb, T_gb_gp, _ = gripper_calib_transforms(gripper_calib)
        ot_ts = np.asarray(root['optitrack/timestamps'][:], dtype=np.float64)
        ot_poses = np.asarray(root['optitrack/pose'][:], dtype=np.float64)

        # Convert OptiTrack poses to optitrack-frame GoPro poses (position + orientation).
        ot_gopro_poses = np.array([transform_optitrack_pose(p, T_gb_rb, T_gb_gp) for p in ot_poses])
        ot_gopro_pos = ot_gopro_poses[:, :3]

        # Collect SLAM poses across all episodes, correcting to the finger/optitrack clock.
        slam_ts_parts: list[np.ndarray] = []
        slam_pos_parts: list[np.ndarray] = []
        slam_quat_parts: list[np.ndarray] = []
        for ep_key in sorted(k for k in root.keys() if k.startswith('episode_')):
            ep_grp = grp(root, ep_key)
            if 'gopro/slam_poses' not in ep_grp:
                continue
            poses = np.asarray(arr(ep_grp, 'gopro/slam_poses')[:], dtype=np.float64)
            gopro_ts = np.asarray(arr(ep_grp, 'timestamps/gopro')[:], dtype=np.float64)

            # Shift GoPro timestamps into the finger (= OptiTrack) clock domain.
            offset = 0.0
            if 'annotations/time_sync' in ep_grp:
                offset = float(
                    ep_grp['annotations/time_sync'].attrs.get('gopro_to_finger_offset_s', 0.0)  # type: ignore[index]
                )
                gopro_ts = gopro_ts - offset

            valid = ~np.isnan(poses[:, 0])
            if valid.any():
                slam_ts_parts.append(gopro_ts[valid])
                slam_pos_parts.append(poses[valid, :3])
                slam_quat_parts.append(poses[valid, 3:])

        if not slam_ts_parts:
            log.warning('No valid SLAM poses found across any episode; storing identity T_ws.')
            self._write_identity(root)
            return

        slam_ts = np.concatenate(slam_ts_parts)
        slam_pos = np.concatenate(slam_pos_parts, axis=0)
        slam_quat = np.concatenate(slam_quat_parts, axis=0)

        # Sort by timestamp (episodes may not be contiguous).
        order = np.argsort(slam_ts)
        slam_ts = slam_ts[order]
        slam_pos = slam_pos[order]
        slam_quat = slam_quat[order]

        # Find the overlap window.
        t_start = max(float(slam_ts.min()), float(ot_ts.min()))
        t_end = min(float(slam_ts.max()), float(ot_ts.max()))

        if t_start >= t_end:
            log.warning(
                f'No time overlap between SLAM ({slam_ts.min():.3f}–{slam_ts.max():.3f}s) '
                f'and OptiTrack ({ot_ts.min():.3f}–{ot_ts.max():.3f}s); storing identity T_ws.'
            )
            self._write_identity(root)
            return

        overlap = (slam_ts >= t_start) & (slam_ts <= t_end)
        slam_ts_ov = slam_ts[overlap]
        slam_pos_ov = slam_pos[overlap]
        slam_quat_ov = slam_quat[overlap]

        log.info(f'Overlap window: {t_end - t_start:.1f}s  ({len(slam_ts_ov)} SLAM poses)')

        # Interpolate OptiTrack positions and orientations at each SLAM timestamp.
        ot_pos_ov = np.column_stack([np.interp(slam_ts_ov, ot_ts, ot_gopro_pos[:, i]) for i in range(3)])
        ot_quat_ov = np.column_stack([np.interp(slam_ts_ov, ot_ts, ot_gopro_poses[:, 3 + i]) for i in range(4)])

        # we want optitrack -> slam so we can make optitrack the parent in a tf tree later.
        R, t = _svd_align(slam_pos_ov, ot_pos_ov)
        rotation_xyzw = Rotation.from_matrix(R).as_quat()

        rms_pos, rms_rot = _calc_rms_errors(R, t, slam_pos_ov, slam_quat_ov, ot_pos_ov, ot_quat_ov)

        log.info(
            f'T_os  translation={t.tolist()}  rotation(xyzw)={rotation_xyzw.tolist()}  '
            f'RMS_pos={rms_pos:.4f} m  RMS_rot={rms_rot:.2f} deg  N={len(slam_pos_ov)}'
        )

        root.attrs['optitrack_to_slam_transform'] = {
            'translation': t.tolist(),
            'rotation': rotation_xyzw.tolist(),
            'rms_pos': rms_pos,
            'rms_rot_deg': rms_rot,
        }

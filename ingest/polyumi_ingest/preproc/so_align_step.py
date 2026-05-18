"""SLAM-to-optitrack alignment preprocessing step."""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from polyumi_ingest.preproc.step_base import PreprocessingStep, register_preprocessing_step
from polyumi_ingest.pzarr.store import arr, grp
from polyumi_ingest.transforms import gripper_calib_transforms, transform_optitrack_pose

log = logging.getLogger(__name__)


def _svd_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for R, t minimising sum ||R @ src_i + t - dst_i||² (Horn's method).

    Args:
        src: (N, 3) positions in source frame (SLAM).
        dst: (N, 3) corresponding positions in destination frame (world).

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation.  world_p ≈ R @ slam_p + t.

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


@register_preprocessing_step(step_number=3, step_name='slam-world-align')
class SlamToWorldAlignStep(PreprocessingStep):
    """
    Compute and store the SLAM-to-world rigid transform T_ws.

    Only runs if there is both slam and optitrack data to align; otherwise stores an identity transform.
    For each episode that has SLAM poses, collects (timestamp, slam_position) pairs
    and applies any time-sync correction so the timestamps align with the OptiTrack
    clock domain.  OptiTrack poses are converted to world-frame GoPro positions via
    the gripper calibration chain.  Within the time overlap window, OptiTrack world
    positions are interpolated to each SLAM timestamp, producing matched point pairs.
    Horn's closed-form SVD method then solves for the rotation R and translation t
    that minimise::

        Σ ‖R · p_slam_i + t − p_world_i‖²

    The result is written to the root zarr attrs as ``slam_to_world_transform``
    with keys ``translation`` (list of 3 floats) and ``rotation`` (list of 4
    floats, xyzw).  The MCAP exporter reads this attr and uses it as the static
    ``world → slam`` frame transform.

    Prerequisites: OptiTrack data in the root group (``optitrack/pose`` and
    ``optitrack/timestamps``) and at least one episode with ``gopro/slam_poses``.
    """

    @staticmethod
    def _write_identity(root: zarr.Group) -> None:
        """Store an identity transform as slam_to_world_transform."""
        root.attrs['slam_to_world_transform'] = {
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0],
        }

    def run_step(self, scene_zarr: pathlib.Path, force: bool = False) -> None:
        """Compute T_ws and write it to the root zarr attrs."""
        root = zarr.open_group(str(scene_zarr), mode='a')

        if 'slam_to_world_transform' in root.attrs and not force:
            log.info('slam_to_world_transform already present; use --force to recompute.')
            return

        if 'optitrack/pose' not in root:
            log.warning('No optitrack data in scene; storing identity T_ws.')
            self._write_identity(root)
            return

        gripper_calib = root.attrs.get('gripper_calib')
        if not isinstance(gripper_calib, dict):
            log.warning('No gripper_calib in scene attrs; storing identity T_ws.')
            self._write_identity(root)
            return

        T_gb_rb, T_gb_gp, T_o_w = gripper_calib_transforms(gripper_calib)
        ot_ts = np.asarray(root['optitrack/timestamps'][:], dtype=np.float64)
        ot_poses = np.asarray(root['optitrack/pose'][:], dtype=np.float64)

        # Convert OptiTrack poses to world-frame GoPro positions (xyz only; SVD is position-based).
        ot_world_pos = np.array([transform_optitrack_pose(p, T_gb_rb, T_gb_gp, T_o_w)[:3] for p in ot_poses])

        # Collect SLAM poses across all episodes, correcting to the finger/optitrack clock.
        slam_ts_parts: list[np.ndarray] = []
        slam_pos_parts: list[np.ndarray] = []
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

        if not slam_ts_parts:
            log.warning('No valid SLAM poses found across any episode; storing identity T_ws.')
            self._write_identity(root)
            return

        slam_ts = np.concatenate(slam_ts_parts)
        slam_pos = np.concatenate(slam_pos_parts, axis=0)

        # Sort by timestamp (episodes may not be contiguous).
        order = np.argsort(slam_ts)
        slam_ts = slam_ts[order]
        slam_pos = slam_pos[order]

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

        log.info(f'Overlap window: {t_end - t_start:.1f}s  ({len(slam_ts_ov)} SLAM poses)')

        # Interpolate OptiTrack world positions at each SLAM timestamp.
        world_pos_ov = np.column_stack([np.interp(slam_ts_ov, ot_ts, ot_world_pos[:, i]) for i in range(3)])

        R, t = _svd_align(slam_pos_ov, world_pos_ov)
        rotation_xyzw = Rotation.from_matrix(R).as_quat()

        residuals = (R @ slam_pos_ov.T).T + t - world_pos_ov  # (N, 3)
        rms_error = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
        log.info(
            f'T_ws  translation={t.tolist()}  rotation(xyzw)={rotation_xyzw.tolist()}  '
            f'RMS={rms_error:.4f} m  N={len(slam_pos_ov)}'
        )

        root.attrs['slam_to_world_transform'] = {
            'translation': t.tolist(),
            'rotation': rotation_xyzw.tolist(),
        }

"""SLAM-to-world alignment preprocessing step."""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import RigidTransform, Rotation

from polyumi_ingest.preproc.step_base import PreprocessingStep, register_preprocessing_step
from polyumi_ingest.pzarr.store import _arr, _grp
from polyumi_ingest.transforms import gripper_calib_transforms, transform_optitrack_pose

log = logging.getLogger(__name__)

_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)


@register_preprocessing_step(step_number=3, step_name='slam-world-align')
class SlamToWorldAlignStep(PreprocessingStep):
    """
    Compute and store the SLAM-to-world rigid transform T_ws.

    For each episode that has SLAM poses, collects (timestamp, slam_pose) pairs
    and applies any time-sync correction so the timestamps align with the
    OptiTrack clock domain.  Finds the overlap window T shared by both
    trajectories, averages the poses on each side (translation via np.mean,
    rotation via Rotation.mean), and solves::

        T_ws = T_wb * T_sb.inv()

    where T_wb is the mean GoPro pose in the world frame (from OptiTrack) and
    T_sb is the mean GoPro pose in the SLAM frame.

    The result is written to the root zarr attrs as ``slam_to_world_transform``
    with keys ``translation`` (list of 3 floats) and ``rotation`` (list of 4
    floats, xyzw).  The MCAP exporter reads this attr and uses it as the static
    ``world → slam`` frame transform.

    Prerequisites: OptiTrack data in the root group (``optitrack/pose`` and
    ``optitrack/timestamps``) and at least one episode with ``gopro/slam_poses``.
    """

    def run_step(self, scene_zarr: pathlib.Path, force: bool = False) -> None:
        """Compute T_ws and write it to the root zarr attrs."""
        root = zarr.open_group(str(scene_zarr), mode='a')

        if 'slam_to_world_transform' in root.attrs and not force:
            log.info('slam_to_world_transform already present; use --force to recompute.')
            return

        if 'optitrack/pose' not in root:
            log.warning('No optitrack data in scene; skipping slam-world alignment.')
            return

        gripper_calib = root.attrs.get('gripper_calib')
        if not isinstance(gripper_calib, dict):
            log.warning('No gripper_calib in scene attrs; skipping slam-world alignment.')
            return

        T_gb_rb, T_gb_gp, T_o_w = gripper_calib_transforms(gripper_calib)
        ot_ts = np.asarray(root['optitrack/timestamps'][:], dtype=np.float64)
        ot_poses = np.asarray(root['optitrack/pose'][:], dtype=np.float64)

        # Collect SLAM poses across all episodes, correcting to the finger/optitrack clock.
        slam_ts_parts: list[np.ndarray] = []
        slam_poses_parts: list[np.ndarray] = []
        for ep_key in sorted(k for k in root.keys() if k.startswith('episode_')):
            ep_grp = _grp(root, ep_key)
            if 'gopro/slam_poses' not in ep_grp:
                continue
            poses = np.asarray(_arr(ep_grp, 'gopro/slam_poses')[:], dtype=np.float64)
            gopro_ts = np.asarray(_arr(ep_grp, 'timestamps/gopro')[:], dtype=np.float64)

            # Shift GoPro timestamps into the finger (= OptiTrack) clock domain.
            if 'annotations/time_sync' in ep_grp:
                offset = float(
                    ep_grp['annotations/time_sync'].attrs.get('gopro_to_finger_offset_s', 0.0)  # type: ignore[index]
                )
                gopro_ts = gopro_ts - offset

            valid = ~np.isnan(poses[:, 0])
            if valid.any():
                slam_ts_parts.append(gopro_ts[valid])
                slam_poses_parts.append(poses[valid])

        if not slam_ts_parts:
            log.warning('No valid SLAM poses found across any episode; skipping slam-world alignment.')
            return

        slam_ts = np.concatenate(slam_ts_parts)
        slam_poses = np.concatenate(slam_poses_parts, axis=0)

        # Find the overlap window T.
        t_start = max(float(slam_ts.min()), float(ot_ts.min()))
        t_end = min(float(slam_ts.max()), float(ot_ts.max()))

        if t_start >= t_end:
            log.warning(
                f'No time overlap between SLAM ({slam_ts.min():.3f}–{slam_ts.max():.3f}s) '
                f'and OptiTrack ({ot_ts.min():.3f}–{ot_ts.max():.3f}s); skipping.'
            )
            return

        slam_poses_T = slam_poses[(slam_ts >= t_start) & (slam_ts <= t_end)]
        ot_poses_T = ot_poses[(ot_ts >= t_start) & (ot_ts <= t_end)]

        if len(slam_poses_T) == 0 or len(ot_poses_T) == 0:
            log.warning('No poses inside the overlap window; skipping slam-world alignment.')
            return

        log.info(
            f'Overlap window: {t_end - t_start:.1f}s  '
            f'({len(slam_poses_T)} SLAM poses, {len(ot_poses_T)} OptiTrack poses)'
        )

        # T_wb: average GoPro pose in world frame (from OptiTrack).
        world_gp = np.array([transform_optitrack_pose(p, T_gb_rb, T_gb_gp, T_o_w) for p in ot_poses_T])
        T_wb = RigidTransform.from_components(
            translation=np.mean(world_gp[:, :3], axis=0),
            rotation=Rotation.from_quat(world_gp[:, 3:]).mean(),
        )

        # T_sb: average GoPro pose in SLAM frame.
        T_sb = RigidTransform.from_components(
            translation=np.mean(slam_poses_T[:, :3], axis=0),
            rotation=Rotation.from_quat(slam_poses_T[:, 3:]).mean(),
        )

        T_ws = T_wb * T_sb.inv()
        log.info(
            f'T_ws  translation={T_ws.translation.tolist()}  '
            f'rotation(xyzw)={T_ws.rotation.as_quat().tolist()}'
        )

        root.attrs['slam_to_world_transform'] = {
            'translation': T_ws.translation.tolist(),
            'rotation': T_ws.rotation.as_quat().tolist(),
        }

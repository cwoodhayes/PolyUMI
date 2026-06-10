"""
Visualize SLAM-to-optitrack alignment for a scene after preprocessing step 3.

Shows three trajectories in the optitrack frame:
  1. OptiTrack ground truth (GoPro position, optitrack frame)
  2. SLAM trajectory after applying the slam->optitrack transform (aligned)
  3. SLAM trajectory before the transform (raw SLAM frame, origin at SLAM origin)

Also draws the SLAM coordinate axes expressed in the optitrack frame.

Usage:
    uv run python ingest/integration/visualize_slam_alignment.py recordings/scene_YYYY-MM-DD_...
"""

import argparse
import logging
import os
import pathlib
import signal
import sys

import matplotlib.pyplot as plt
import numpy as np
import zarr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from rich.logging import RichHandler
from scipy.spatial.transform import Rotation

from polyumi_ingest.config import load_gripper_calib
from polyumi_ingest.pzarr.scene_files import SceneFiles
from polyumi_ingest.transforms import gripper_calib_transforms, transform_optitrack_pose

log = logging.getLogger(__name__)


def _load_optitrack_gopro(root: zarr.Group) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, positions_xyz) for the optitrack GoPro trajectory."""
    gripper_calib = dict(root.attrs.get('gripper_calib', {})) or load_gripper_calib()
    T_gb_rb, T_gb_gp, _ = gripper_calib_transforms(gripper_calib)
    ot_ts = np.asarray(root['optitrack/timestamps'][:], dtype=np.float64)
    ot_poses = np.asarray(root['optitrack/pose'][:], dtype=np.float64)
    ot_gopro = np.array([transform_optitrack_pose(p, T_gb_rb, T_gb_gp) for p in ot_poses])
    return ot_ts, ot_gopro  # (N,7)


def _load_slam_poses(root: zarr.Group) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (timestamps, poses_xyz) across all episodes, time-sync corrected.

    Returns only the valid (non-NaN) rows, sorted by timestamp.
    """
    ts_parts: list[np.ndarray] = []
    pos_parts: list[np.ndarray] = []
    for ep_key in sorted(k for k in root.keys() if k.startswith('episode_')):
        ep = root[ep_key]
        if 'gopro/slam_poses' not in ep:
            continue
        poses = np.asarray(ep['gopro/slam_poses'][:], dtype=np.float64)
        gopro_ts = np.asarray(ep['timestamps/gopro'][:], dtype=np.float64)
        offset = 0.0
        if 'annotations/time_sync' in ep:
            offset = float(ep['annotations/time_sync'].attrs.get('gopro_to_finger_offset_s', 0.0))
        gopro_ts = gopro_ts - offset
        valid = ~np.isnan(poses[:, 0])
        if valid.any():
            ts_parts.append(gopro_ts[valid])
            pos_parts.append(poses[valid, :3])
    if not ts_parts:
        return np.empty(0), np.empty((0, 3))
    ts = np.concatenate(ts_parts)
    pos = np.concatenate(pos_parts, axis=0)
    order = np.argsort(ts)
    return ts[order], pos[order]


def _draw_axes(ax: plt.Axes, origin: np.ndarray, R: np.ndarray, scale: float, label: str) -> None:
    """Draw RGB xyz axes at origin with orientation R (3x3, columns = x/y/z dirs)."""
    colors = ['red', 'green', 'blue']
    labels = [f'{label} x', f'{label} y', f'{label} z']
    for i, (color, lbl) in enumerate(zip(colors, labels)):
        end = origin + scale * R[:, i]
        ax.quiver(
            origin[0], origin[1], origin[2],
            R[0, i], R[1, i], R[2, i],
            length=scale, color=color, linewidth=2, arrow_length_ratio=0.2, label=lbl,
        )


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
        format='%(message)s',
        handlers=[RichHandler(show_time=True, show_level=True, show_path=False, rich_tracebacks=True)],
    )
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene', type=pathlib.Path, help='Scene directory or scene.zarr path.')
    parser.add_argument('--axis-scale', type=float, default=0.3,
                        help='Length of drawn coordinate axes (metres, default 0.3).')
    args = parser.parse_args()

    scene_zarr = SceneFiles.resolve_zarr_path(args.scene)
    if not scene_zarr.exists():
        print(f'error: no scene.zarr found at {args.scene}', file=sys.stderr)
        sys.exit(1)

    root = zarr.open_group(str(scene_zarr), mode='r')

    if 'optitrack_to_slam_transform' not in root.attrs:
        print('error: no optitrack_to_slam_transform in scene.zarr (run pingest pp 3 first)', file=sys.stderr)
        sys.exit(1)

    # --- load data ---
    ot_ts, ot_gopro = _load_optitrack_gopro(root)  # (N,7) in optitrack frame
    slam_ts, slam_pos = _load_slam_poses(root)       # (M,3) in SLAM frame

    tf = root.attrs['optitrack_to_slam_transform']
    t_vec = np.array(tf['translation'], dtype=np.float64)   # slam frame -> optitrack frame
    R_mat = Rotation.from_quat(tf['rotation']).as_matrix()  # same

    # Transform: optitrack_pos = R @ slam_pos + t
    slam_pos_aligned = (R_mat @ slam_pos.T).T + t_vec

    # SLAM trajectory before transform: keep in SLAM frame but shift so its centroid
    # sits near the optitrack centroid (just for visual "unaligned" reference).
    # We do NOT apply R or t — we only translate by the centroid offset so it's
    # visible in the same plot without the rotation applied.
    slam_centroid = slam_pos.mean(axis=0)
    ot_centroid = ot_gopro[:, :3].mean(axis=0)
    slam_pos_unaligned = slam_pos - slam_centroid + ot_centroid  # shifted, not rotated

    # SLAM frame axes in optitrack frame: columns of R_mat are where SLAM x/y/z point
    # The SLAM frame origin in optitrack coords = t_vec (when slam_pos = 0)
    slam_origin_in_ot = t_vec
    slam_R_in_ot = R_mat  # columns are SLAM axes expressed in optitrack frame

    rms_pos = tf.get('rms_pos', float('nan'))
    rms_rot = tf.get('rms_rot_deg', float('nan'))
    scene_name = scene_zarr.parent.name

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(ot_gopro[:, 0], ot_gopro[:, 1], ot_gopro[:, 2],
            color='royalblue', linewidth=1.2, alpha=0.85, label='OptiTrack (ground truth)')

    ax.plot(slam_pos_aligned[:, 0], slam_pos_aligned[:, 1], slam_pos_aligned[:, 2],
            color='darkorange', linewidth=1.0, alpha=0.85, label='SLAM aligned (R·p + t)')

    ax.plot(slam_pos_unaligned[:, 0], slam_pos_unaligned[:, 1], slam_pos_unaligned[:, 2],
            color='gray', linewidth=0.8, alpha=0.5, linestyle='--', label='SLAM unaligned (centroid-shifted)')

    # Optitrack frame axes at its centroid (origin of optitrack frame shown as identity axes)
    _draw_axes(ax, ot_centroid, np.eye(3), args.axis_scale, 'OT')

    # SLAM frame axes at its origin expressed in optitrack frame
    _draw_axes(ax, slam_origin_in_ot, slam_R_in_ot, args.axis_scale, 'SLAM')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    title = (
        f'{scene_name}\n'
        f'RMS pos = {rms_pos:.3f} m   RMS rot = {rms_rot:.1f}°\n'
        f'N_optitrack={len(ot_ts)}  N_slam={len(slam_ts)}'
    )
    ax.set_title(title, fontsize=9)

    # Deduplicate quiver labels (mpl adds one per arrow)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, int] = {}
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = 1
            dedup_h.append(h)
            dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, fontsize=8, loc='upper left')

    ax.set_aspect('equal')
    fig.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

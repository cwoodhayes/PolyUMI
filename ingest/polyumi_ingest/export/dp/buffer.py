"""
Export a pzarr scene to a diffusion-policy ReplayBuffer zarr.

Produces the flat layout that diffusion_policy's DexNexDataset expects::

    <output>.zarr/
      meta/episode_ends            (n_episodes,) int64, cumulative step counts
      data/img      (T,256,256,3)  float32 in [0,1], gopro frames at 10 Hz
      data/state    (T,8)          float32  [x,y,z,qx,qy,qz,qw, gripper_m]
      data/action   (T,8)          float32  (copy of state; training computes relative traj)
      data/reward   (T,)           float32  zeros
      data/not_done (T,)           float32  1.0, last step of each episode 0.0

The store is written ``zarr_format=2`` (as the rest of pzarr is) so diffusion_policy,
which is pinned to the zarr v2 API, can read it without importing anything from here.
"""

from __future__ import annotations

import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import imagecodecs.numcodecs  # noqa: F401 — registers imagecodecs_jpegxl with numcodecs
import numpy as np
import zarr
from numcodecs import Blosc

from polyumi_ingest.pzarr.scene_files import SceneFiles
from polyumi_ingest.pzarr.store import arr, grp

log = logging.getLogger('export.dp')

HZ = 10
RESOLUTION = 256
_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)


def _nearest_idx(sorted_ts: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the nearest value in ascending ``sorted_ts`` for each ``query`` time."""
    idx = np.searchsorted(sorted_ts, query)
    idx = np.clip(idx, 1, len(sorted_ts) - 1)
    closer_left = (query - sorted_ts[idx - 1]) <= (sorted_ts[idx] - query)
    idx = idx - closer_left
    return np.clip(idx, 0, len(sorted_ts) - 1)


def _decode_resized_frames(frames_arr: zarr.Array, gidx: np.ndarray) -> np.ndarray:
    """Decode the selected gopro frames and resize+normalize to (T,RES,RES,3) float32 [0,1]."""

    def one(i: int) -> np.ndarray:
        frame = np.asarray(frames_arr[int(i)])  # (H,W,3) uint8 RGB
        resized = cv2.resize(frame, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    with ThreadPoolExecutor() as executor:
        frames = list(executor.map(one, gidx))
    return np.stack(frames, axis=0)


def _append(data_grp: zarr.Group, arrays: dict[str, np.ndarray]) -> None:
    """Append each array along axis 0, creating resizable v2 arrays on first use."""
    for key, value in arrays.items():
        value = np.asarray(value)
        t = value.shape[0]
        if key not in data_grp:
            chunks = (1,) + value.shape[1:] if value.ndim >= 3 else value.shape
            data_grp.zeros(
                name=key,
                shape=value.shape,
                chunks=chunks,
                dtype=value.dtype,
                compressor=_BLOSC,
                zarr_format=2,
            )
            arr(data_grp, key)[:] = value
        else:
            a = arr(data_grp, key)
            old = a.shape[0]
            a.resize((old + t,) + a.shape[1:])
            a[old:] = value


def _export_episode(
    ep: zarr.Group,
    root: zarr.Group,
    data_grp: zarr.Group,
    episode_key: str,
    pose_source: str,
) -> int:
    """Resample one episode to HZ and append it to ``data_grp``. Returns the step count T."""
    # GoPro runs on its own clock; bring it into the finger (Pi) clock domain.
    offset = float(grp(ep, 'annotations/time_sync').attrs['gopro_to_finger_offset_s'])
    gopro_ts = arr(ep, 'timestamps/gopro')[:] - offset
    finger_ts = arr(ep, 'timestamps/finger')[:]
    opti_ts = arr(root, 'optitrack/timestamps')[:]

    # Window: start when the last stream comes online, end when the first drops out.
    t_start = max(gopro_ts[0], finger_ts[0], opti_ts[0])
    t_end = min(gopro_ts[-1], finger_ts[-1], opti_ts[-1])
    if t_end <= t_start:
        raise RuntimeError(f'{episode_key}: empty overlap window [{t_start}, {t_end}]')
    target_ts = np.arange(t_start, t_end, 1.0 / HZ)
    t = len(target_ts)

    # img, gripper width, and slam poses all live on the gopro frame grid → one index.
    gidx = _nearest_idx(gopro_ts, target_ts)
    img = _decode_resized_frames(arr(ep, 'gopro/frames'), gidx)

    if pose_source == 'optitrack':
        pose = arr(root, 'optitrack/pose')[:][_nearest_idx(opti_ts, target_ts)]
    elif pose_source == 'slam':
        pose = arr(ep, 'gopro/slam_poses')[:][gidx]
    else:
        raise ValueError(f"pose_source must be 'optitrack' or 'slam', got {pose_source!r}")
    pose = np.asarray(pose, dtype=np.float32)

    gripper = arr(ep, 'annotations/gripper_width/width_m')[:][gidx].astype(np.float32)

    if np.isnan(pose).any():
        raise RuntimeError(
            f'{episode_key}: pose source {pose_source!r} contains NaN over the window '
            f'({int(np.isnan(pose[:, 0]).sum())}/{t} steps). Refusing to write.'
        )
    if np.isnan(gripper).any():
        raise RuntimeError(
            f'{episode_key}: gripper width contains NaN over the window '
            f'({int(np.isnan(gripper).sum())}/{t} steps). Refusing to write.'
        )

    state = np.concatenate([pose, gripper[:, None]], axis=1).astype(np.float32)
    not_done = np.ones(t, dtype=np.float32)
    not_done[-1] = 0.0
    _append(
        data_grp,
        {
            'img': img,
            'state': state,
            'action': state.copy(),
            'reward': np.zeros(t, dtype=np.float32),
            'not_done': not_done,
        },
    )
    log.info(f'  {episode_key}: {t} steps @ {HZ} Hz (pose={pose_source})')
    return t


def export_scene_to_dp(
    scene_path: pathlib.Path,
    output_path: pathlib.Path,
    pose_source: str = 'optitrack',
) -> int:
    """Export EPISODE sessions of a pzarr scene to a diffusion-policy ReplayBuffer zarr.

    Returns the number of episodes written. MAPPING sessions are skipped.
    """
    zarr_path = SceneFiles.resolve_zarr_path(scene_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

    root = zarr.open_group(str(zarr_path), mode='r')
    n_episodes = int(root.attrs.get('n_episodes', 0))

    out = zarr.open_group(str(output_path), mode='w', zarr_format=2)
    meta = out.create_group('meta')
    data = out.create_group('data')

    episode_ends: list[int] = []
    total = 0
    for i in range(n_episodes):
        ep_key = f'episode_{i}'
        if ep_key not in root:
            log.warning(f'{ep_key} not found in {zarr_path.name}, skipping.')
            continue
        ep = zarr.open_group(str(zarr_path / ep_key), mode='r')
        if ep.attrs.get('session_type') == 'MAPPING':
            log.info(f'  {ep_key}: MAPPING session, skipping.')
            continue
        total += _export_episode(ep, root, data, ep_key, pose_source)
        episode_ends.append(total)

    meta.create_array('episode_ends', data=np.array(episode_ends, dtype=np.int64), compressor=_BLOSC)
    log.info(f'Wrote {len(episode_ends)} episode(s), {total} steps → {output_path}')
    return len(episode_ends)

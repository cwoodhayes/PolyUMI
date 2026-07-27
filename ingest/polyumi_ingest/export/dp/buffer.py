"""
Export a pzarr scene to a UMI-format ReplayBuffer (``.zarr.zip``).

The layout matches ``universal_manipulation_interface``'s ``ReplayBuffer`` so that
``UmiDataset`` reads it directly — the key *names* are load-bearing (the sampler counts
robots via ``key.endswith('eef_pos')``, picks Slerp vs linear interp via ``'rot' in key``,
and ``get_normalizer`` raises on any low-dim key it can't name-match)::

    <output>.zarr.zip
      meta/episode_ends                 (n_episodes,) int64, cumulative step counts
      data/camera0_rgb                  (T,224,224,3) uint8  Blosc(zstd), chunks (1,224,224,3)
      data/robot0_eef_pos               (T,3)  float32  hand-frame position
      data/robot0_eef_rot_axis_angle    (T,3)  float32  hand-frame rotation as a rotvec
      data/robot0_gripper_width         (T,1)  float32  metres
      data/robot0_demo_start_pose       (T,6)  float32  episode's first [pos, rotvec], repeated
      data/robot0_demo_end_pose         (T,6)  float32  episode's last  [pos, rotvec], repeated

Deliberately absent:

* ``action`` — ``sampler.py`` synthesises it from
  ``[eef_pos, eef_rot_axis_angle, gripper_width]`` when the key is missing, so writing it
  would be redundant (and would have to be kept in lock-step with the obs keys).
* ``robot0_eef_rot_axis_angle_wrt_start`` — ``UmiDataset`` derives it at load time from
  ``demo_start_pose``. It is in ``shape_meta`` but must *not* be in the store.
* tactile (piezo / finger camera) — out of scope for the visuomotor policy.

Poses come from each episode's ``eef/pose`` (preprocessing step 5), already on the canonical
**hand** body frame and on the GoPro frame grid. Quaternion → rotvec is the only pose
transform here. The world frame is left as-is: it cancels out of the relative trajectory the
policy trains on, whereas the body frame does not — which is why step 5 exists. See
``EefPoseStep`` and ``transforms.retarget_body_frame``.

Frames are exported at the **native GoPro rate** (~59.94 Hz), not down-sampled here. UMI's
dataset assumes uniform Δt and sets the effective observation rate itself via
``obs_down_sample_steps`` in the task config, so storing raw keeps that knob meaningful and
lets the obs rate change without re-exporting.

Images use **Blosc**, not JpegXl, on purpose. The training container pins Python 3.9 with
``imagecodecs==2023.9.18``, whose JpegXl codec cannot parse the config that our Python-3.13
``imagecodecs`` (2026.x) writes (``bitspersample``, ``squeeze``, ``usecontainer``, …), and no
single ``imagecodecs`` release supports both interpreters. Blosc is in ``numcodecs`` core, so
its config is byte-identical across both stacks; it is lossless, decodes faster than JpegXl,
and only costs disk.

GoPro frames are decoded on demand from each episode's ``gopro.mp4`` sidecar (via
``video_helpers.open_gopro_frames``) — the pzarr no longer stores a ``gopro/frames`` array.
The resize onto the 224x224 ``camera0_rgb`` grid goes through the shared
``camera_preproc.resize_camera0_rgb`` contract so it stays identical to what the inference
node feeds the policy.

The store is written ``zarr_format=2`` so the (v2-pinned) UMI zarr can read it, then packed
into a ``.zarr.zip`` because ``UmiDataset`` opens its dataset through ``zarr.ZipStore``.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile
import zipfile

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation

from polyumi_ingest.camera_preproc import CAMERA0_RGB_RESOLUTION, resize_camera0_rgb
from polyumi_ingest.manifests import SceneManifest
from polyumi_ingest.pzarr.scene_files import SceneFiles
from polyumi_ingest.pzarr.store import arr, grp
from polyumi_ingest.video_helpers import GoproMp4Frames, open_gopro_frames

log = logging.getLogger('export.dp')

#: Output image resolution, matching UMI's ``camera0_rgb`` shape ``[3, 224, 224]``.
RESOLUTION = CAMERA0_RGB_RESOLUTION
#: Single robot arm; UMI's multi-robot keying still applies with one ``robot0``.
_ROBOT = 'robot0'
_CAMERA = 'camera0_rgb'

_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
_TIME_CHUNK = 1024

#: Warn if the largest inter-frame gap in the exported span exceeds this multiple of the
#: median frame period. Frames are stored as-is (no resampling), so a dropped frame becomes a
#: single step that silently violates UMI's uniform-Δt assumption. One is tolerable; many
#: suggests a recording problem.
_GAP_WARN_FACTOR = 5.0


def _measure_rate(gopro_ts: np.ndarray) -> float:
    """Native frame rate (Hz) from the median inter-frame period of GoPro timestamps."""
    if len(gopro_ts) < 2:
        raise RuntimeError(f'Need at least 2 GoPro timestamps to measure rate, got {len(gopro_ts)}')
    return 1.0 / float(np.median(np.diff(gopro_ts)))


def _longest_valid_span(valid: np.ndarray) -> tuple[int, int]:
    """
    Return the inclusive [start, end] index range of the longest run of True in ``valid``.

    An episode can lose its pose source mid-way (SLAM tracking loss shows up as NaN rows in
    ``eef/pose``). Dropping scattered rows would break the temporal continuity UMI's fixed-rate
    sampler assumes, so we export the single longest gap-free stretch instead.
    """
    best_start, best_len = 0, 0
    run_start = None
    for i, ok in enumerate(valid):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None
    if run_start is not None and len(valid) - run_start > best_len:
        best_start, best_len = run_start, len(valid) - run_start
    if best_len == 0:
        raise RuntimeError('no valid (non-NaN) pose/gripper rows in episode')
    return best_start, best_start + best_len - 1


def _decode_resized_frames(frames_arr: GoproMp4Frames, gidx: np.ndarray) -> np.ndarray:
    """Decode the selected GoPro frames and resize to (T,RES,RES,3) uint8 (channel-last)."""
    # gidx is a contiguous ascending range, so decode sequentially — the mp4 reader is a
    # forward-only single VideoCapture (not thread-safe), and H.264 decode is inherently
    # sequential anyway. resize_camera0_rgb is the shared export/inference transform.
    frames = [resize_camera0_rgb(np.asarray(frames_arr[int(i)])) for i in gidx]
    return np.stack(frames, axis=0).astype(np.uint8)


def _append(data_grp: zarr.Group, arrays: dict[str, np.ndarray]) -> None:
    """Append each array along axis 0, creating resizable v2 arrays on first use."""
    for key, value in arrays.items():
        value = np.asarray(value)
        t = value.shape[0]
        if key not in data_grp:
            if value.ndim >= 3:
                chunks = (1,) + value.shape[1:]
            else:
                chunks = (min(t, _TIME_CHUNK),) + value.shape[1:]
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


def _export_episode(ep: zarr.Group, data_grp: zarr.Group, episode_key: str, scene_zarr: pathlib.Path) -> int:
    """Export one episode's longest valid span at native rate and append it. Returns step count T."""
    if 'eef/pose' not in ep:
        raise RuntimeError(f'{episode_key}: no eef/pose — run preprocessing step 5 (eef-pose) before exporting.')
    gopro_ts = np.asarray(arr(ep, 'timestamps/gopro')[:], dtype=np.float64)
    pose = np.asarray(arr(ep, 'eef/pose')[:], dtype=np.float64)  # (N,7) [xyz, quat] hand frame
    gripper = np.asarray(arr(ep, 'annotations/gripper_width/width_m')[:], dtype=np.float64)
    frames = open_gopro_frames(ep, scene_zarr)

    n = len(gopro_ts)
    if not (len(pose) == len(gripper) == frames.shape[0] == n):
        raise RuntimeError(
            f'{episode_key}: GoPro-grid arrays disagree in length — '
            f'gopro_ts={n}, eef/pose={len(pose)}, gripper={len(gripper)}, frames={frames.shape[0]}. '
            f'eef/pose and gripper_width must be on the GoPro grid (steps 4 and 5).'
        )

    # Valid window: the longest gap-free run where both pose and gripper are non-NaN. This is
    # what replaces the old cross-stream overlap window — no finger/optitrack clock needed,
    # since frames, pose, and gripper already share the GoPro grid.
    valid = ~np.isnan(pose).any(axis=1) & ~np.isnan(gripper)
    i0, i1 = _longest_valid_span(valid)
    n_dropped = n - (i1 - i0 + 1)
    if n_dropped:
        log.warning(
            f'  {episode_key}: kept longest valid span [{i0}, {i1}] of {n} frames '
            f'({n_dropped} dropped as NaN/outside span).'
        )

    # Export the frames as recorded — no resampling. UMI's dataset assumes uniform Δt and sets
    # the observation rate itself via obs_down_sample_steps, so the exporter's job is just to
    # hand over the raw native-rate stream. GoPro records at a steady ~59.94 Hz locally; a large
    # inter-frame gap means a dropped frame, which would be treated as a single step, so warn.
    span_ts = gopro_ts[i0 : i1 + 1]
    rate = _measure_rate(span_ts)
    max_gap = float(np.max(np.diff(span_ts)))
    if max_gap > _GAP_WARN_FACTOR / rate:
        log.warning(
            f'  {episode_key}: largest inter-frame gap {max_gap * 1e3:.0f} ms is '
            f'>{_GAP_WARN_FACTOR:g}x the {1e3 / rate:.1f} ms median — a dropped frame is stored '
            f'as one step, breaking the uniform-rate assumption there.'
        )

    gidx = np.arange(i0, i1 + 1)
    t = len(gidx)

    pos = pose[gidx, :3].astype(np.float32)
    rotvec = Rotation.from_quat(pose[gidx, 3:]).as_rotvec().astype(np.float32)
    tcp6 = np.concatenate([pos, rotvec], axis=1)  # (T,6) [pos, rotvec] — UMI's tcp_pose

    _append(
        data_grp,
        {
            _CAMERA: _decode_resized_frames(frames, gidx),
            f'{_ROBOT}_eef_pos': pos,
            f'{_ROBOT}_eef_rot_axis_angle': rotvec,
            f'{_ROBOT}_gripper_width': gripper[gidx, None].astype(np.float32),
            f'{_ROBOT}_demo_start_pose': np.broadcast_to(tcp6[0], (t, 6)).copy(),
            f'{_ROBOT}_demo_end_pose': np.broadcast_to(tcp6[-1], (t, 6)).copy(),
        },
    )
    source = grp(ep, 'eef').attrs.get('source', 'unknown')
    log.info(f'  {episode_key}: {t} steps @ {rate:.2f} Hz (pose={source})')
    return t


def _zip_zarr_dir(zarr_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    """Pack the *contents* of a directory zarr into a ``.zarr.zip`` readable by zarr.ZipStore."""
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_STORED) as zf:
        for path in sorted(zarr_dir.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(zarr_dir).as_posix())


def _append_scene_episodes(scene_path: pathlib.Path, data_grp: zarr.Group, episode_ends: list[int], total: int) -> int:
    """Append every EPISODE session of one scene onto ``data_grp``, returning the new running total."""
    zarr_path = SceneFiles.resolve_zarr_path(scene_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

    # zarr_path.name is always the literal 'scene.zarr', identical across every scene; use the
    # scene directory's own name so multi-scene exports can tell episodes from different scenes
    # apart in logs/errors (otherwise every scene logs as e.g. 'scene.zarr/episode_0').
    scene_label = zarr_path.parent.name

    # scene.json (not the pzarr) is the canonical home of the unusable-episode marker set from
    # the catalog UI, so it's checked here rather than baked into pzarr at build time.
    # zarr_path.parent is the scene root regardless of whether scene_path itself was given as
    # the scene root or as a direct .zarr path (see resolve_zarr_path).
    manifest = SceneManifest.from_scene_dir(zarr_path.parent)
    unusable_dirs = set(manifest.unusable_episodes) if manifest else set()

    root = zarr.open_group(str(zarr_path), mode='r')
    n_episodes = int(root.attrs.get('n_episodes', 0))
    for i in range(n_episodes):
        ep_key = f'episode_{i}'
        if ep_key not in root:
            log.warning(f'{ep_key} not found in {scene_label}, skipping.')
            continue
        ep = zarr.open_group(str(zarr_path / ep_key), mode='r')
        if ep.attrs.get('session_type') == 'MAPPING':
            log.info(f'  {scene_label}/{ep_key}: MAPPING session, skipping.')
            continue
        if ep.attrs.get('session_dir') in unusable_dirs:
            log.info(f'  {scene_label}/{ep_key}: marked unusable, skipping.')
            continue
        total += _export_episode(ep, data_grp, f'{scene_label}/{ep_key}', zarr_path)
        episode_ends.append(total)
    return total


def export_scene_to_dp(scene_path: pathlib.Path, output_path: pathlib.Path) -> int:
    """
    Export EPISODE sessions of a pzarr scene to a UMI-format ``.zarr.zip`` ReplayBuffer.

    Poses come from ``eef/pose`` (preprocessing step 5), which has already resolved the
    optitrack-vs-slam source choice and put the trajectory on the hand frame.

    Returns the number of episodes written. MAPPING sessions and episodes marked unusable
    in ``scene.json`` are skipped.
    """
    return export_scenes_to_dp([scene_path], output_path)


def export_scenes_to_dp(scene_paths: list[pathlib.Path], output_path: pathlib.Path) -> int:
    """
    Export EPISODE sessions from one or more pzarr scenes into a single UMI ``.zarr.zip``.

    Each scene's episodes are appended in the given order, with ``episode_ends`` accumulating
    across the whole list — a multi-scene dataset is indistinguishable from a single big scene
    to ``UmiDataset``, which only ever sees one buffer. MAPPING sessions and episodes marked
    unusable in ``scene.json`` are skipped scene by scene, same as the single-scene exporter.
    Poses come from ``eef/pose`` (preprocessing step 5).

    Returns the total number of episodes written across all scenes.
    """
    if not scene_paths:
        raise ValueError('No scenes given to export.')

    with tempfile.TemporaryDirectory(prefix='dp_export_') as tmp:
        build_dir = pathlib.Path(tmp) / 'buffer.zarr'
        out = zarr.open_group(str(build_dir), mode='w', zarr_format=2)
        meta = out.create_group('meta')
        data = out.create_group('data')

        episode_ends: list[int] = []
        total = 0
        for scene_path in scene_paths:
            total = _append_scene_episodes(scene_path, data, episode_ends, total)

        if not episode_ends:
            raise RuntimeError('no EPISODE sessions to export across the given scene(s).')

        meta.create_array('episode_ends', data=np.array(episode_ends, dtype=np.int64), compressor=_BLOSC)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _zip_zarr_dir(build_dir, output_path)

    log.info(f'Wrote {len(episode_ends)} episode(s) from {len(scene_paths)} scene(s), {total} steps → {output_path}')
    return len(episode_ends)

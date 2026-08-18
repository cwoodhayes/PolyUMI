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
      data/robot0_gripper_width         (T,1)  float32  metres of opening from fully closed
      data/robot0_demo_start_pose       (T,6)  float32  episode's first [pos, rotvec], repeated
      data/robot0_demo_end_pose         (T,6)  float32  episode's last  [pos, rotvec], repeated

Deliberately absent:

* ``action`` — ``sampler.py`` synthesises it from
  ``[eef_pos, eef_rot_axis_angle, gripper_width]`` when the key is missing, so writing it
  would be redundant (and would have to be kept in lock-step with the obs keys).
* ``robot0_eef_rot_axis_angle_wrt_start`` — ``UmiDataset`` derives it at load time from
  ``demo_start_pose``. It is in ``shape_meta`` but must *not* be in the store.
* tactile (piezo / finger camera) — out of scope for the visuomotor policy.

Poses come from one of each episode's ``eef/pose_<source>`` arrays (preprocessing step 5 writes
one per available source — ``optitrack`` and/or ``slam``), already on the canonical **hand**
body frame and on the GoPro frame grid. Step 5 no longer picks a single winner; *this* exporter
does, per episode: an episode's ``eef.attrs['default_source']`` (OptiTrack if present, else
SLAM) unless overridden by ``scene.json``'s ``pose_source_overrides`` (keyed by session
directory name, same pattern as ``unusable_episodes``) — see ``resolve_pose_source``. Changing
the source is therefore a re-*export*, not a re-preprocess. Quaternion → rotvec is the only pose
transform here. The world frame is left as-is: it cancels out of the relative trajectory the
policy trains on, whereas the body frame does not — which is why step 5 exists. See
``EefPoseStep`` and ``transforms.retarget_body_frame``.

Each exported episode's resolved source is recorded as **provenance**: alongside the returned
episode count, ``export_scenes_to_dp``/``export_scene_to_dp`` return a list of per-episode
provenance dicts (scene, session, episode, source, world_frame, n_steps, segment, frame_range,
frame_stride), which callers can persist externally (see ``main.py``'s ``<output>.provenance.json`` sidecar and the
catalog's ``DatasetManifest``). The same information is also written into the ``.zarr.zip``
itself, as ``meta.attrs['pose_provenance']`` (the full list) and ``meta.attrs['episode_pose_source']``
(just the per-episode source strings, aligned with ``episode_ends``) — ``UmiDataset`` only ever
reads ``meta/episode_ends``, so these extra attrs are inert cargo it ignores.

Steps are the frames SLAM was **fed** — every ``localization_frame_stride``-th GoPro frame, so
~30 Hz at the current stride of 2, not the 59.94 Hz the camera records at. Poses exist only on
that grid (nothing is interpolated), and Δt stays uniform, which is all UMI's dataset requires;
it sets the observation rate from there via ``obs_down_sample_steps`` in the task config. That
knob and this stride are coupled: halving the stored rate must halve ``obs_down_sample_steps``
or the policy trains on a different Δt than it runs at.

One session can produce **several episodes**. Where the pose source has no pose — SLAM lost
tracking — the session is split into the contiguous runs either side, each exported as its own
episode, and runs shorter than ``MIN_SEGMENT_STEPS`` are dropped. Bridging a gap would put a
step of the wrong duration inside an episode, which the fixed-rate sampler cannot see; splitting
keeps every episode honest. This follows the upstream UMI repo's
``scripts_slam_pipeline/06_generate_dataset_plan.py``.

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

from polyumi_ingest import quality
from polyumi_ingest.camera_preproc import CAMERA0_RGB_RESOLUTION, resize_camera0_rgb
from polyumi_ingest.config import load_closed_width_m, load_open_width_m
from polyumi_ingest.manifests import SceneManifest
from polyumi_ingest.preproc import available_preprocessing_steps, preprocessing_steps_done
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

#: Shortest run of valid steps worth emitting as its own episode. Matches upstream UMI's
#: ``--min_episode_length`` default. Anything shorter can't supply a full observation +
#: action horizon, so it would only ever be padding.
MIN_SEGMENT_STEPS = 24


def _episode_frame_stride(ep: zarr.Group) -> int:
    """
    Return the frame decimation SLAM ran at, which is the grid poses exist on.

    ``localization_frame_stride`` (step 2) feeds the localizer every Nth GoPro frame, so the
    other frames have no pose at all. Defaults to 1 for stores predating the attr, where the
    localizer saw every frame and the fed grid is the full grid.
    """
    if 'annotations/slam' not in ep:
        return 1
    return max(1, int(grp(ep, 'annotations/slam').attrs.get('frame_stride', 1)))


def _measure_rate(gopro_ts: np.ndarray) -> float:
    """Native frame rate (Hz) from the median inter-frame period of GoPro timestamps."""
    if len(gopro_ts) < 2:
        raise RuntimeError(f'Need at least 2 GoPro timestamps to measure rate, got {len(gopro_ts)}')
    return 1.0 / float(np.median(np.diff(gopro_ts)))


def _valid_segments(valid: np.ndarray, min_steps: int) -> list[tuple[int, int]]:
    """
    Split a validity mask into contiguous True runs, dropping those shorter than ``min_steps``.

    Returns inclusive ``[start, end]`` index pairs into ``valid``.

    One episode per run, rather than the single longest run we used to keep: a demo whose pose
    source drops out in the middle is two usable demonstrations either side of the hole, not one
    truncated one. This mirrors upstream UMI's ``get_bool_segments`` in
    the upstream UMI repo's ``scripts_slam_pipeline/06_generate_dataset_plan.py``. Runs shorter
    than ``min_steps`` are too short to sample a
    horizon from and are discarded rather than emitted as degenerate episodes.

    Splitting rather than bridging is the whole point: UMI's fixed-rate sampler assumes uniform
    Δt *within* an episode, which a gap would silently violate.
    """
    segments: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, ok in enumerate(valid):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            segments.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        segments.append((run_start, len(valid) - 1))
    return [(s, e) for s, e in segments if e - s + 1 >= min_steps]


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


def _auto_unusable_reasons_for_episode(ep: zarr.Group) -> list[str]:
    """
    Threshold-derived reasons to exclude ``ep``; empty list means keep it.

    Thin adapter over ``polyumi_ingest.quality.auto_unusable_reasons`` that pulls its
    inputs off the episode group: the SLAM metrics written by step 2, the hand-frame
    pose jump written by step 5, and whether OptiTrack is among step 5's
    ``available_sources`` (in which case the episode's poses don't come from SLAM and
    the SLAM-derived checks don't apply).

    Missing groups mean "nothing to judge" — an episode whose preprocessing hasn't run
    isn't excluded here; ``resolve_pose_source`` raises on that separately.
    """
    if 'annotations' not in ep or 'slam' not in grp(ep, 'annotations'):
        return []
    slam_attrs = dict(grp(grp(ep, 'annotations'), 'slam').attrs)
    has_optitrack = False
    if 'eef' in ep:
        eef = grp(ep, 'eef')
        has_optitrack = 'optitrack' in list(eef.attrs.get('available_sources', []))
        # Measured on the SLAM trajectory specifically; an OptiTrack episode is exempt from
        # the SLAM checks anyway, so there is no second source to reconcile here.
        if 'pose_slam' in eef:
            jump = eef['pose_slam'].attrs.get('max_pose_jump_m')
            if jump is not None:
                slam_attrs['max_pose_jump_m'] = jump
    return quality.auto_unusable_reasons(slam_attrs, has_optitrack=has_optitrack)


def resolve_pose_source(ep: zarr.Group, episode_key: str, override: str | None) -> str:
    """
    Resolve which pose source ``episode_key`` should export from.

    ``override`` (from ``scene.json``'s ``pose_source_overrides``, keyed by session directory
    name) wins when set; otherwise falls back to ``eef.attrs['default_source']`` (OptiTrack if
    the episode has it, else SLAM — see ``EefPoseStep``). Raises if step 5 hasn't run at all, or
    if the resolved source's array isn't one the episode actually has.
    """
    if 'eef' not in ep:
        raise RuntimeError(f'{episode_key}: no eef group — run preprocessing step 5 (eef-pose) before exporting.')
    eef_grp = grp(ep, 'eef')
    available = list(eef_grp.attrs.get('available_sources', []))
    if override is not None:
        source = override
        if source not in available:
            raise RuntimeError(
                f'{episode_key}: pose_source_overrides requests {source!r}, but this episode only '
                f'has {available} (run `pingest pp 5 --force` if it should have more).'
            )
    else:
        source = eef_grp.attrs.get('default_source')
        if source is None or f'pose_{source}' not in eef_grp:
            raise RuntimeError(
                f'{episode_key}: eef has no default_source / eef/pose_{source} — '
                f're-run preprocessing step 5 (eef-pose) before exporting.'
            )
    return source


def _export_episode(
    ep: zarr.Group,
    data_grp: zarr.Group,
    episode_key: str,
    scene_zarr: pathlib.Path,
    pose_source: str,
    closed_width_m: float,
    open_width_m: float,
    min_segment_steps: int = MIN_SEGMENT_STEPS,
) -> list[tuple[int, dict]]:
    """
    Export one session as one DP episode per contiguous valid segment.

    Returns ``[(T, provenance), ...]`` — possibly empty if nothing survived, one entry per
    segment appended to ``data_grp``.
    """
    array_name = f'pose_{pose_source}'
    if 'eef' not in ep or array_name not in grp(ep, 'eef'):
        raise RuntimeError(
            f'{episode_key}: no eef/{array_name} — run preprocessing step 5 (eef-pose) before exporting.'
        )
    pose_attrs = arr(ep, f'eef/{array_name}').attrs
    gopro_ts = np.asarray(arr(ep, 'timestamps/gopro')[:], dtype=np.float64)
    pose = np.asarray(arr(ep, f'eef/{array_name}')[:], dtype=np.float64)  # (N,7) [xyz, quat] hand frame
    # Raw ArUco tag separation, converted here to opening-from-closed. The subtraction lives in
    # the exporter rather than in step 4 on purpose: step 4 is the expensive pass (ArUco over every
    # frame), so keeping the pzarr annotation calibration-independent means re-deriving closed width
    # costs a re-export, not a re-detect. UMI applies it at the same stage, in
    # the upstream UMI repo's scripts_slam_pipeline/06_generate_dataset_plan.py, not at detection.
    #
    # Clamped at both ends, as UMI does: its get_gripper_calibration_interpolator
    # (umi/common/interpolation_util.py, upstream) builds an interp1d over
    # [min_width, max_width] with bounds_error=False, fill_value=(x[0], x[-1]), so detections
    # outside the calibrated range saturate rather than escape.
    #
    # The bottom clamp is unavoidable: closed width is a percentile, so ~1% of detections sit
    # below it by construction and would export negative.
    #
    # The top clamp was deliberately omitted until 2026-08-18, on the reasoning that it would
    # make open_mm load-bearing and hide a demo opening wider than the calibration recording —
    # "information, not error". That holds for a compliant mechanism; it does not hold for this
    # one. The handheld gripper has a hard stop, so a separation above open_mm is not a wider
    # demonstration, it is a misread tag. red_trapezoid_mug_v4 carried a single sample at
    # 208.8 mm of opening — 253.4 mm of raw tag separation on a gripper that stops at 132.33 —
    # and because UMI normalizes this channel by min/max, that one frame in 22640 set the top of
    # the range and squeezed the real 0-86 mm signal into the bottom 41% of it.
    #
    # The old concern survives as the warning below rather than as missing clamping: a
    # miscalibrated open_mm now announces itself instead of silently widening the range.
    max_opening_m = open_width_m - closed_width_m
    gripper = np.asarray(arr(ep, 'annotations/gripper_width/width_m')[:], dtype=np.float64)
    gripper = np.maximum(gripper - closed_width_m, 0.0)
    n_over = int((gripper > max_opening_m).sum())
    if n_over:
        log.warning(
            f'  {episode_key}: {n_over}/{len(gripper)} gripper width(s) exceed the calibrated '
            f'stroke ({max_opening_m * 1e3:.2f} mm, max seen {gripper.max() * 1e3:.2f} mm); '
            f'clamping. A few are misread tags; many mean open_mm in gripper_calib.yaml is '
            f'wrong — re-derive it with `pingest calibrate-gripper`.'
        )
    gripper = np.minimum(gripper, max_opening_m)
    frames = open_gopro_frames(ep, scene_zarr)

    n = len(gopro_ts)
    if not (len(pose) == len(gripper) == frames.shape[0] == n):
        raise RuntimeError(
            f'{episode_key}: GoPro-grid arrays disagree in length — '
            f'gopro_ts={n}, eef/{array_name}={len(pose)}, gripper={len(gripper)}, frames={frames.shape[0]}. '
            f'eef/{array_name} and gripper_width must be on the GoPro grid (steps 4 and 5).'
        )

    # The step grid is the frames SLAM was actually fed, not every GoPro frame. Poses only
    # exist on that grid (nothing is interpolated any more), so exporting the full grid would
    # make every other row NaN. Δt is uniform at stride/rate, which is what UMI's sampler needs.
    stride = _episode_frame_stride(ep)
    steps = np.arange(0, n, stride)

    # Mask out the idle prefix before the sync chirp: the operator waits for the chirp, so those
    # frames shouldn't train the policy. Masking (rather than nudging a start index) means the
    # exported span is exactly the span quality.py gates on, and it composes with segmentation.
    chirp_end_s = None
    if 'annotations/time_sync' in ep:
        chirp_end_s = grp(ep, 'annotations/time_sync').attrs.get('gopro_chirp_end_s')
    if chirp_end_s is None:
        # enforce_preprocessing (checked in _append_scene_episodes) guarantees step 1 ran, but
        # not that this specific marker exists (e.g. an older pzarr predating it) — stay non-fatal.
        log.warning(
            f'  {episode_key}: no gopro_chirp_end_s annotation (missing, or an older pzarr '
            f'predating this marker); exporting without start trim.'
        )
    else:
        trimmed = steps[gopro_ts[steps] >= float(chirp_end_s)]
        if len(trimmed) < min_segment_steps:
            # A chirp end past (or nearly past) the episode means the detection was wrong, not
            # that the demo is one long idle prefix. Dropping the episode on that evidence would
            # let one bad correlation peak silently delete real data, so distrust the marker.
            log.warning(
                f'  {episode_key}: chirp end leaves only {len(trimmed)} of {len(steps)} steps — '
                f'likely a bad chirp detection; not trimming.'
            )
        else:
            log.info(f'  {episode_key}: trimmed {len(steps) - len(trimmed)} step(s) before chirp end.')
            steps = trimmed

    valid = ~np.isnan(pose[steps]).any(axis=1) & ~np.isnan(gripper[steps])
    segments = _valid_segments(valid, min_segment_steps)
    if not segments:
        log.warning(
            f'  {episode_key}: no valid run of >={min_segment_steps} steps in {len(steps)} fed frames; skipping.'
        )
        return []
    if len(segments) > 1:
        log.info(f'  {episode_key}: pose gaps split this session into {len(segments)} episodes.')

    results: list[tuple[int, dict]] = []
    for seg_i, (s0, s1) in enumerate(segments):
        gidx = steps[s0 : s1 + 1]
        t = len(gidx)

        # No resampling: the stored Δt *is* the dataset's, and UMI sets the observation rate
        # from it via obs_down_sample_steps. A gap much larger than the median means a dropped
        # frame, which would be stored as one ordinary step and silently bend the time base.
        span_ts = gopro_ts[gidx]
        rate = _measure_rate(span_ts)
        max_gap = float(np.max(np.diff(span_ts)))
        if max_gap > _GAP_WARN_FACTOR / rate:
            log.warning(
                f'  {episode_key}: largest inter-frame gap {max_gap * 1e3:.0f} ms is '
                f'>{_GAP_WARN_FACTOR:g}x the {1e3 / rate:.1f} ms median — a dropped frame is stored '
                f'as one step, breaking the uniform-rate assumption there.'
            )

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
        seg_label = f' segment {seg_i}' if len(segments) > 1 else ''
        log.info(f'  {episode_key}{seg_label}: {t} steps @ {rate:.2f} Hz (pose={pose_source})')
        # 'episode'/'scene'/'session' are filled in by the caller (_append_scene_episodes), which
        # knows the plain episode key ('episode_0') and scene/session names — episode_key here is
        # the combined 'scene_label/episode_0' string used only for log/error messages.
        results.append(
            (
                t,
                {
                    'source': pose_source,
                    'world_frame': pose_attrs.get('world_frame'),
                    'n_steps': t,
                    'segment': seg_i,
                    'frame_range': [int(gidx[0]), int(gidx[-1])],
                    'frame_stride': stride,
                },
            )
        )
    return results


def _zip_zarr_dir(zarr_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    """Pack the *contents* of a directory zarr into a ``.zarr.zip`` readable by zarr.ZipStore."""
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_STORED) as zf:
        for path in sorted(zarr_dir.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(zarr_dir).as_posix())


def _check_preprocessing_complete(root: zarr.Group, scene_label: str) -> None:
    """
    Raise if the scene is missing any registered preprocessing step.

    The DP export reads the outputs of the whole pipeline (``eef/pose`` from step 5,
    gripper width from step 4, and the chirp-end marker from step 1), so an incompletely
    preprocessed scene would silently export a partial/untrimmed dataset. Callers can bypass
    this with ``enforce_preprocessing=False``.
    """
    done = set(preprocessing_steps_done(root))
    required = {cls.step_number for cls in available_preprocessing_steps()}
    missing = sorted(required - done)
    if missing:
        raise RuntimeError(
            f'{scene_label}: preprocessing steps {missing} not complete (done={sorted(done)}). '
            f'Run `pingest pp` first, or export with enforce_preprocessing=False to skip this check.'
        )


def _append_scene_episodes(
    scene_path: pathlib.Path,
    data_grp: zarr.Group,
    episode_ends: list[int],
    total: int,
    provenance: list[dict],
    closed_width_m: float,
    open_width_m: float,
    enforce_preprocessing: bool = True,
    min_segment_steps: int = MIN_SEGMENT_STEPS,
) -> int:
    """Append every EPISODE session of one scene onto ``data_grp``, returning the new running total."""
    zarr_path = SceneFiles.resolve_zarr_path(scene_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

    # zarr_path.name is always the literal 'scene.zarr', identical across every scene; use the
    # scene directory's own name so multi-scene exports can tell episodes from different scenes
    # apart in logs/errors (otherwise every scene logs as e.g. 'scene.zarr/episode_0').
    scene_label = zarr_path.parent.name

    # scene.json (not the pzarr) is the canonical home of the unusable-episode marker set and
    # the pose-source override, both from the catalog UI, so it's checked here rather than
    # baked into pzarr at build time. zarr_path.parent is the scene root regardless of whether
    # scene_path itself was given as the scene root or as a direct .zarr path (see
    # resolve_zarr_path).
    manifest = SceneManifest.from_scene_dir(zarr_path.parent)
    unusable_dirs = set(manifest.unusable_episodes) if manifest else set()
    pose_source_overrides = manifest.pose_source_overrides if manifest else {}

    root = zarr.open_group(str(zarr_path), mode='r')
    if enforce_preprocessing:
        _check_preprocessing_complete(root, scene_label)
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
        session_dir = ep.attrs.get('session_dir')
        if session_dir in unusable_dirs:
            log.info(f'  {scene_label}/{ep_key}: marked unusable, skipping.')
            continue
        # Threshold-derived exclusion, on top of the explicit scene.json set above.
        # Same function the catalog UI calls, so what the UI shows as excluded is
        # exactly what's skipped here. Thresholds: config/quality_thresholds.yaml.
        auto_reasons = _auto_unusable_reasons_for_episode(ep)
        if auto_reasons:
            log.info(f'  {scene_label}/{ep_key}: unusable ({"; ".join(auto_reasons)}), skipping.')
            continue
        pose_source = resolve_pose_source(ep, f'{scene_label}/{ep_key}', pose_source_overrides.get(session_dir))
        # One session can yield several episodes — a pose gap splits it into the usable runs
        # either side — or none, if nothing survived the chirp trim and the length floor.
        for t, ep_provenance in _export_episode(
            ep,
            data_grp,
            f'{scene_label}/{ep_key}',
            zarr_path,
            pose_source,
            min_segment_steps=min_segment_steps,
            closed_width_m=closed_width_m,
            open_width_m=open_width_m,
        ):
            total += t
            episode_ends.append(total)
            provenance.append({'scene': scene_label, 'session': session_dir, 'episode': ep_key, **ep_provenance})
    return total


def export_scene_to_dp(
    scene_path: pathlib.Path,
    output_path: pathlib.Path,
    enforce_preprocessing: bool = True,
    min_segment_steps: int = MIN_SEGMENT_STEPS,
) -> tuple[int, list[dict]]:
    """
    Export EPISODE sessions of a pzarr scene to a UMI-format ``.zarr.zip`` ReplayBuffer.

    Poses come from ``eef/pose_<source>`` (preprocessing step 5, which writes one array per
    available source); *this* function resolves the optitrack-vs-slam choice per episode — see
    ``resolve_pose_source`` — and puts the trajectory on the hand frame.

    When ``enforce_preprocessing`` is True (default), the scene must have every registered
    preprocessing step complete or export raises. This does not by itself guarantee the
    chirp-end marker is present — a scene preprocessed before that marker was added can still
    be missing it, in which case the start trim is skipped non-fatally (see ``_export_episode``).
    Set ``enforce_preprocessing`` False to export a partially preprocessed scene instead.

    Returns ``(n_episodes, provenance)`` — the episode count and a per-episode pose-source
    provenance list (also embedded in the ``.zarr.zip``'s ``meta`` attrs; see module docstring).
    MAPPING sessions and episodes marked unusable in ``scene.json`` are skipped.
    """
    return export_scenes_to_dp(
        [scene_path],
        output_path,
        enforce_preprocessing=enforce_preprocessing,
        min_segment_steps=min_segment_steps,
    )


def export_scenes_to_dp(
    scene_paths: list[pathlib.Path],
    output_path: pathlib.Path,
    enforce_preprocessing: bool = True,
    min_segment_steps: int = MIN_SEGMENT_STEPS,
) -> tuple[int, list[dict]]:
    """
    Export EPISODE sessions from one or more pzarr scenes into a single UMI ``.zarr.zip``.

    Each scene's episodes are appended in the given order, with ``episode_ends`` accumulating
    across the whole list — a multi-scene dataset is indistinguishable from a single big scene
    to ``UmiDataset``, which only ever sees one buffer. MAPPING sessions and episodes marked
    unusable in ``scene.json`` are skipped scene by scene, same as the single-scene exporter.
    Poses come from ``eef/pose_<source>`` (preprocessing step 5); the per-episode source choice
    is resolved here (see ``resolve_pose_source``). ``enforce_preprocessing`` (default True)
    requires every registered preprocessing step to be complete on each scene.

    Returns ``(n_episodes, provenance)`` — the total episode count across all scenes and a
    per-episode pose-source provenance list, in export order (also embedded in the ``.zarr.zip``
    as ``meta.attrs['pose_provenance']`` / ``meta.attrs['episode_pose_source']``).
    """
    if not scene_paths:
        raise ValueError('No scenes given to export.')

    # Resolved once for the whole buffer, not per scene: every episode in one ReplayBuffer has to
    # be in the same width units, and re-reading the config mid-export would silently mix two
    # calibrations if the file changed underneath.
    closed_width_m = load_closed_width_m()
    open_width_m = load_open_width_m()
    log.info(
        f'Gripper widths exported as opening-from-closed (closed width = {closed_width_m * 1000:.2f} mm), '
        f'clamped to the calibrated stroke ({(open_width_m - closed_width_m) * 1000:.2f} mm).'
    )

    with tempfile.TemporaryDirectory(prefix='dp_export_') as tmp:
        build_dir = pathlib.Path(tmp) / 'buffer.zarr'
        out = zarr.open_group(str(build_dir), mode='w', zarr_format=2)
        meta = out.create_group('meta')
        data = out.create_group('data')

        episode_ends: list[int] = []
        provenance: list[dict] = []
        total = 0
        for scene_path in scene_paths:
            total = _append_scene_episodes(
                scene_path,
                data,
                episode_ends,
                total,
                provenance,
                enforce_preprocessing=enforce_preprocessing,
                min_segment_steps=min_segment_steps,
                closed_width_m=closed_width_m,
                open_width_m=open_width_m,
            )

        # Every episode in one buffer must share a time base: UmiDataset reads a single
        # episode_ends array and assumes one uniform Δt across all of it, so a mix of strides
        # would train on two different notions of "one step" with nothing recording which.
        strides = {p['frame_stride'] for p in provenance if 'frame_stride' in p}
        if len(strides) > 1:
            raise RuntimeError(
                f'Refusing to write a mixed-rate buffer: episodes were localized at frame strides '
                f'{sorted(strides)}, so their steps span different durations. Re-run `pingest pp 2 '
                f'--force` on the odd scenes so every episode shares one stride.'
            )

        if not episode_ends:
            raise RuntimeError('no EPISODE sessions to export across the given scene(s).')

        meta.create_array('episode_ends', data=np.array(episode_ends, dtype=np.int64), compressor=_BLOSC)
        # Inert cargo for UmiDataset (it only reads meta/episode_ends) — lets anyone who opens
        # the .zarr.zip directly see which pose source produced each episode without needing the
        # external manifest/sidecar too. See module docstring.
        meta.attrs['pose_provenance'] = provenance
        meta.attrs['episode_pose_source'] = [p['source'] for p in provenance]
        # What ``robot0_gripper_width`` means, carried with the data. Without it a buffer is
        # indistinguishable from one exported before the subtraction existed, and the inference
        # side needs to know which convention a checkpoint was trained under to pick its offset.
        meta.attrs['gripper_closed_width_m'] = float(closed_width_m)
        meta.attrs['gripper_open_width_m'] = float(open_width_m)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _zip_zarr_dir(build_dir, output_path)

    log.info(f'Wrote {len(episode_ends)} episode(s) from {len(scene_paths)} scene(s), {total} steps → {output_path}')
    return len(episode_ends), provenance

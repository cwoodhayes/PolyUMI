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
provenance dicts (scene, session, episode, source, world_frame, n_steps, n_interp_filled), which
callers can persist externally (see ``main.py``'s ``<output>.provenance.json`` sidecar and the
catalog's ``DatasetManifest``). The same information is also written into the ``.zarr.zip``
itself, as ``meta.attrs['pose_provenance']`` (the full list) and ``meta.attrs['episode_pose_source']``
(just the per-episode source strings, aligned with ``episode_ends``) — ``UmiDataset`` only ever
reads ``meta/episode_ends``, so these extra attrs are inert cargo it ignores.

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

from polyumi_ingest import quality
from polyumi_ingest.camera_preproc import CAMERA0_RGB_RESOLUTION, resize_camera0_rgb
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


def _auto_unusable_reasons_for_episode(ep: zarr.Group) -> list[str]:
    """
    Threshold-derived reasons to exclude ``ep``; empty list means keep it.

    Thin adapter over ``polyumi_ingest.quality.auto_unusable_reasons`` that pulls the
    two inputs off the episode group: the SLAM metrics written by step 2, and whether
    OptiTrack is among step 5's ``available_sources`` (in which case the episode's
    poses don't come from SLAM and the SLAM-derived checks don't apply).

    Missing groups mean "nothing to judge" — an episode whose preprocessing hasn't run
    isn't excluded here; ``resolve_pose_source`` raises on that separately.
    """
    if 'annotations' not in ep or 'slam' not in grp(ep, 'annotations'):
        return []
    slam_attrs = dict(grp(grp(ep, 'annotations'), 'slam').attrs)
    has_optitrack = False
    if 'eef' in ep:
        has_optitrack = 'optitrack' in list(grp(ep, 'eef').attrs.get('available_sources', []))
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
    ep: zarr.Group, data_grp: zarr.Group, episode_key: str, scene_zarr: pathlib.Path, pose_source: str
) -> tuple[int, dict]:
    """Export one episode's longest valid span at native rate and append it. Returns (T, provenance)."""
    array_name = f'pose_{pose_source}'
    if 'eef' not in ep or array_name not in grp(ep, 'eef'):
        raise RuntimeError(
            f'{episode_key}: no eef/{array_name} — run preprocessing step 5 (eef-pose) before exporting.'
        )
    pose_attrs = arr(ep, f'eef/{array_name}').attrs
    gopro_ts = np.asarray(arr(ep, 'timestamps/gopro')[:], dtype=np.float64)
    pose = np.asarray(arr(ep, f'eef/{array_name}')[:], dtype=np.float64)  # (N,7) [xyz, quat] hand frame
    gripper = np.asarray(arr(ep, 'annotations/gripper_width/width_m')[:], dtype=np.float64)
    frames = open_gopro_frames(ep, scene_zarr)

    n = len(gopro_ts)
    if not (len(pose) == len(gripper) == frames.shape[0] == n):
        raise RuntimeError(
            f'{episode_key}: GoPro-grid arrays disagree in length — '
            f'gopro_ts={n}, eef/{array_name}={len(pose)}, gripper={len(gripper)}, frames={frames.shape[0]}. '
            f'eef/{array_name} and gripper_width must be on the GoPro grid (steps 4 and 5).'
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

    # Advance the start past the sync chirp: the operator waits for the chirp before starting,
    # so the frames up to gopro_chirp_end_s are an idle prefix that shouldn't train the policy.
    # gopro_chirp_end_s (step 1, chirp-time-sync) is on the same clock as gopro_ts. Moving i0
    # forward within the contiguous valid run keeps every remaining row valid.
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
        first_after = int(np.searchsorted(gopro_ts, float(chirp_end_s), side='left'))
        # _measure_rate needs >=2 frames, so require at least 2 remaining after the trim
        # (first_after <= i1 - 1); this also covers chirp end landing past the whole span.
        if first_after >= i1:
            log.warning(
                f'  {episode_key}: chirp end leaves fewer than 2 valid frames — '
                f'likely a bad chirp detection; not trimming.'
            )
        elif first_after > i0:
            log.info(
                f'  {episode_key}: trimmed {first_after - i0} leading frame(s) '
                f'(~{(first_after - i0) / _measure_rate(gopro_ts):.2f}s) before chirp end.'
            )
            i0 = first_after

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
    log.info(f'  {episode_key}: {t} steps @ {rate:.2f} Hz (pose={pose_source})')
    # 'episode'/'scene'/'session' are filled in by the caller (_append_scene_episodes), which
    # knows the plain episode key ('episode_0') and scene/session names — episode_key here is
    # the combined 'scene_label/episode_0' string used only for log/error messages.
    provenance = {
        'source': pose_source,
        'world_frame': pose_attrs.get('world_frame'),
        'n_steps': t,
        'n_interp_filled': pose_attrs.get('n_interp_filled', 0),
    }
    return t, provenance


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
    enforce_preprocessing: bool = True,
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
        t, ep_provenance = _export_episode(ep, data_grp, f'{scene_label}/{ep_key}', zarr_path, pose_source)
        total += t
        episode_ends.append(total)
        provenance.append({'scene': scene_label, 'session': session_dir, 'episode': ep_key, **ep_provenance})
    return total


def export_scene_to_dp(
    scene_path: pathlib.Path, output_path: pathlib.Path, enforce_preprocessing: bool = True
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
    return export_scenes_to_dp([scene_path], output_path, enforce_preprocessing=enforce_preprocessing)


def export_scenes_to_dp(
    scene_paths: list[pathlib.Path], output_path: pathlib.Path, enforce_preprocessing: bool = True
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
                scene_path, data, episode_ends, total, provenance, enforce_preprocessing=enforce_preprocessing
            )

        if not episode_ends:
            raise RuntimeError('no EPISODE sessions to export across the given scene(s).')

        meta.create_array('episode_ends', data=np.array(episode_ends, dtype=np.int64), compressor=_BLOSC)
        # Inert cargo for UmiDataset (it only reads meta/episode_ends) — lets anyone who opens
        # the .zarr.zip directly see which pose source produced each episode without needing the
        # external manifest/sidecar too. See module docstring.
        meta.attrs['pose_provenance'] = provenance
        meta.attrs['episode_pose_source'] = [p['source'] for p in provenance]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _zip_zarr_dir(build_dir, output_path)

    log.info(f'Wrote {len(episode_ends)} episode(s) from {len(scene_paths)} scene(s), {total} steps → {output_path}')
    return len(episode_ends), provenance

"""ORB-SLAM3 Monocular-Inertial preprocessing step."""

from __future__ import annotations

import csv
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time

import imagecodecs.numcodecs  # noqa: F401 — registers imagecodecs_jpegxl with numcodecs
import numpy as np
import zarr
from numcodecs import Blosc

from polyumi_ingest.config import SLAM_CONFIG_YAML, load_slam_config
from polyumi_ingest.episode_status import Episode, SceneContext
from polyumi_ingest.preproc.step_base import (
    PreprocessingStep,
    register_preprocessing_step,
)
from polyumi_ingest.pzarr.scene_files import resolve_gopro_mp4
from polyumi_ingest.pzarr.store import arr, grp

log = logging.getLogger(__name__)

_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

# Marker string used in the settings YAML to flag values that need calibration.
_PLACEHOLDER_MARKER = 'CALIBRATE_ME'

# Repo-root-relative default install path for the ORB-SLAM3 fork — the git
# submodule at external/ORB_SLAM3_PolyUMI.  Set ORB_SLAM3_DIR in the env to
# override (useful if you're working out-of-tree).
_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
_DEFAULT_ORB_SLAM3_DIR = _REPO_ROOT / 'external' / 'ORB_SLAM3_PolyUMI'

_DEFAULT_SETTINGS_YAML = _DEFAULT_ORB_SLAM3_DIR / 'Examples' / 'Monocular-Inertial' / 'gopro_hero12_slam.yaml'

# Gripper mask: the camera-rigid hardware blanked out before tracking. Unlike the settings
# YAML this lives in the repo proper, not the submodule, because it is ours — it describes
# the PolyUMI gripper, and a `git submodule update` must not be able to clobber it.
# Hand-drawn against a temporal-median frame; white (non-zero) = discarded, black = kept.
_SLAM_MASK_PNG = _REPO_ROOT / 'ingest' / 'config' / 'slam_mask.png'

# How far (as a fraction of a frame period) a trajectory row's timestamp may sit from the
# frame its index maps to. Rows are indexed directly, so this is a consistency assertion on
# the decimation stride rather than a matching tolerance — not a tuning knob, hence not in
# slam.yaml.
_TRAJ_TOLERANCE_FRAC = 0.5


def _require_slam_setting(key: str) -> int:
    """
    Read one required tunable from ``config/slam.yaml``, raising if it isn't there.

    Deliberately has no default to fall back on. These values decide what ORB-SLAM3 is fed,
    they are stamped into every episode, and DP export refuses to mix two of them in one
    buffer — so a default that silently disagreed with the checked-in config would split a
    corpus across incompatible time bases rather than fail.
    """
    config = load_slam_config()
    try:
        return int(config[key])
    except KeyError:
        raise KeyError(
            f'{SLAM_CONFIG_YAML} has no {key!r} entry. It controls how much data the SLAM '
            f'step is fed and has no safe default; add it to the config file.'
        ) from None


def _export_telemetry_json(
    gyro: np.ndarray,
    gyro_ts: np.ndarray,
    accl: np.ndarray,
    accl_ts: np.ndarray,
    t_ref: float,
    json_path: pathlib.Path,
) -> None:
    """
    Write a GoPro GPMF-style telemetry JSON.

    The mono_inertial_gopro_vi binary expects::

        {"1": {"streams": {"ACCL": {"samples": [{"value":[z,x,y],"cts":ms}, ...]},
                            "GYRO": {"samples": [...]}}}}

    Axis order is preserved as raw GoPro [z,x,y]; the C++ binary reorders to
    body [x,y,z] via ``value[1], value[2], value[0]``. Timestamps (``cts``)
    are ms relative to ``t_ref`` (the first video frame's UTC time), so the
    IMU and video share a common time origin.

    Accelerometer samples are linearly interpolated onto the gyro timestamps
    because the upstream binary iterates ACCL and GYRO independently and
    assumes a 1:1 mapping between the two streams.
    """
    assert np.all(np.diff(accl_ts) > 0), 'accl_ts must be strictly monotonically increasing'
    accl_interp = np.column_stack([np.interp(gyro_ts, accl_ts, accl[:, j]) for j in range(3)])

    cts_ms = (gyro_ts - t_ref) * 1000.0

    accl_samples = [
        {
            'value': [float(accl_interp[i, 0]), float(accl_interp[i, 1]), float(accl_interp[i, 2])],
            'cts': float(cts_ms[i]),
        }
        for i in range(len(gyro_ts))
    ]
    gyro_samples = [
        {
            'value': [float(gyro[i, 0]), float(gyro[i, 1]), float(gyro[i, 2])],
            'cts': float(cts_ms[i]),
        }
        for i in range(len(gyro_ts))
    ]

    blob = {
        '1': {
            'streams': {
                'ACCL': {'samples': accl_samples},
                'GYRO': {'samples': gyro_samples},
            }
        }
    }
    with open(json_path, 'w') as fh:
        json.dump(blob, fh)


def _export_episode(
    ep_grp: zarr.Group,
    tmp_dir: pathlib.Path,
    gopro_mp4: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, np.ndarray]:
    """
    Export an episode's IMU to a telemetry JSON in ``tmp_dir``.

    ``gopro_mp4`` is passed directly to the ORB-SLAM3 binary as the video
    input; no re-encoding is performed.

    Returns (video_path, json_path, frame_ts) where ``frame_ts`` is the
    per-frame UTC timestamp array (needed downstream to reconcile the
    C++ trajectory output back onto the original frame indices).
    """
    gopro_ts = np.asarray(arr(ep_grp, 'timestamps/gopro')[:], dtype=np.float64)
    if len(gopro_ts) < 2:
        raise RuntimeError(f'Episode has fewer than 2 frames ({len(gopro_ts)})')
    video_path = gopro_mp4
    log.info(f'  Using original gopro.mp4: {gopro_mp4} ({len(gopro_ts)} frames)')

    gyro = np.asarray(arr(ep_grp, 'gopro/gyro')[:], dtype=np.float64)
    gyro_ts = np.asarray(arr(ep_grp, 'timestamps/gopro_gyro')[:], dtype=np.float64)
    accl = np.asarray(arr(ep_grp, 'gopro/accl')[:], dtype=np.float64)
    accl_ts = np.asarray(arr(ep_grp, 'timestamps/gopro_accl')[:], dtype=np.float64)

    json_path = tmp_dir / 'telemetry.json'
    _export_telemetry_json(gyro, gyro_ts, accl, accl_ts, float(gopro_ts[0]), json_path)
    log.info(f'  Exported {len(gyro_ts)} IMU samples to {json_path}')

    return video_path, json_path, gopro_ts


#: Settings keys that scale linearly with image size.  ``Camera.k1..k4`` deliberately
#: do NOT appear here: KannalaBrandt's distortion polynomial acts on the incidence
#: angle theta, which is dimensionless and independent of resolution.
_RESOLUTION_SCALED_KEYS = (
    'Camera.fx',
    'Camera.fy',
    'Camera.cx',
    'Camera.cy',
    'Camera.width',
    'Camera.height',
)


def _downsample_settings(content: str, res_div: int, fps_div: int) -> str:
    """
    Rewrite a settings YAML for reduced resolution and/or frame rate.

    Derived from the canonical YAML rather than kept as a second checked-in file:
    ``gopro_hero12_slam.yaml`` is the calibration source of truth, and a
    hand-maintained half-res copy would silently drift the next time the camera is
    recalibrated.

    ``res_div`` scales the intrinsics and image size (see
    ``_RESOLUTION_SCALED_KEYS``); ``fps_div`` divides ``Camera.fps``, which
    ORB-SLAM3 turns into its keyframe-insertion window (``mMaxFrames``), so a
    decimated run must declare its true rate or keyframes are inserted on the wrong
    cadence.  Both default to 1 (no change) at the call sites that don't downsample.
    """
    if res_div == 1 and fps_div == 1:
        return content
    out = []
    for line in content.splitlines(keepends=True):
        m = re.match(r'^(\s*)([\w.]+)\s*:\s*([-\d.eE+]+)(.*)$', line)
        if m:
            indent, key, value, rest = m.groups()
            if res_div != 1 and key in _RESOLUTION_SCALED_KEYS:
                line = f'{indent}{key}: {float(value) / res_div:.10f}{rest}\n'
            elif fps_div != 1 and key == 'Camera.fps':
                line = f'{indent}{key}: {float(value) / fps_div:.6f}  # /{fps_div} from {value}{rest}\n'
        out.append(line)
    return ''.join(out)


def _make_temp_settings_yaml(
    src: pathlib.Path,
    tmp_dir: pathlib.Path,
    save_atlas: pathlib.Path | None = None,
    load_atlas: pathlib.Path | None = None,
    viewer: bool = False,
    res_div: int = 1,
    fps_div: int = 1,
    mask_png: pathlib.Path | None = _SLAM_MASK_PNG,
) -> pathlib.Path:
    """
    Copy ``src`` settings YAML to ``tmp_dir`` with atlas paths appended.

    ORB-SLAM3 reads atlas save/load paths from the YAML
    (``System.SaveAtlasToFile`` / ``System.LoadAtlasFromFile``); the binary
    has no CLI flag for them. We inject the right key here so the canonical
    config file stays untouched.

    ``mask_png`` rides along the same way, for the same reason: both PolyUMI binaries
    already overload their positional argv, so a new CLI arg would be ambiguous. It is
    the gripper mask — the camera-rigid hardware (fingers, ArUco tags, LEDs, mirrors)
    that must be blanked before tracking or it destroys both two-view init and
    relocalization. Belongs with the camera settings because it describes the rig.

    It defaults to the shipped mask rather than to None so that masking is what you get by
    forgetting, not what you lose by it: an unmasked run doesn't crash, it quietly produces a
    map nothing relocalizes against, and the debug viewer forgetting it is exactly how you
    end up debugging a run that behaves unlike the one you're trying to reproduce. Pass
    ``mask_png=None`` to deliberately track unmasked.

    ``res_div`` / ``fps_div`` optionally downsample the camera settings first; see
    ``_downsample_settings``.
    """
    content = _downsample_settings(src.read_text(), res_div, fps_div)
    if not content.endswith('\n'):
        content += '\n'
    content += f'\nSystem.Viewer: {1 if viewer else 0}\n'
    if save_atlas is not None:
        content += f'\nSystem.SaveAtlasToFile: "{save_atlas}"\n'
    if load_atlas is not None:
        content += f'\nSystem.LoadAtlasFromFile: "{load_atlas}"\n'
    if mask_png is not None:
        content += f'\nMask.Path: "{mask_png}"\n'
    dst = tmp_dir / 'settings.yaml'
    dst.write_text(content)
    return dst


def _parse_trajectory_csv(traj_path: pathlib.Path, frame_ts: np.ndarray, frame_stride: int = 1) -> np.ndarray:
    """
    Parse the localizer's CSV trajectory onto the full GoPro frame grid.

    ``System::SaveTrajectoryCSV`` writes one row per frame the binary was *fed*, in order,
    with a header and an explicit lost flag::

        frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w

    Row ``k`` is therefore the k-th fed frame, i.e. original frame ``k * frame_stride`` --
    a direct index, which is why nothing here matches timestamps. (The EuRoC writer omitted
    lost frames entirely, so every row had to be matched back by time within half a frame
    period, and a mis-estimated anchor shifted poses silently.)

    Poses are in the **camera optical frame**: unlike ``SaveTrajectoryEuRoC``, whose inertial
    branch composes ``mTbc`` and reports the IMU body pose, this writer reports ``Twc``
    directly. See the note in ``mono_inertial_gopro_vi_localize.cc``.

    The ``timestamp`` column is video seconds, so it cross-checks the stride assumption: if
    the binary decimated differently than we think, the times won't line up and this raises
    rather than silently returning poses attached to the wrong frames.

    Returns ``poses`` shaped (N,7) float64 ``[x,y,z, qx,qy,qz,qw]``, all-NaN where lost or
    never fed.
    """
    n = len(frame_ts)
    poses = np.full((n, 7), np.nan, dtype=np.float64)
    if n < 2:
        return poses

    t_ref = float(frame_ts[0])
    tolerance = _TRAJ_TOLERANCE_FRAC * float(np.median(np.diff(frame_ts)))

    n_tracked = n_lost = n_dropped = 0
    with open(traj_path, newline='') as fh:
        for row in csv.DictReader(fh):
            idx = int(row['frame_idx']) * frame_stride
            if idx >= n:
                # The decoder can overrun the end of the mp4 and feed empty frames; those
                # rows describe frames that don't exist on our grid.
                n_dropped += 1
                continue
            if row['is_lost'].strip().lower() == 'true':
                n_lost += 1
                continue

            drift = abs((t_ref + float(row['timestamp'])) - float(frame_ts[idx]))
            if drift > tolerance:
                raise RuntimeError(
                    f'Trajectory row {row["frame_idx"]} claims video t={row["timestamp"]}s, which is '
                    f'{drift * 1000:.1f}ms from frame {idx} (tolerance {tolerance * 1000:.1f}ms). '
                    f'The localizer did not decimate at the stride {frame_stride} assumed here, so '
                    f'every pose would land on the wrong frame.'
                )

            poses[idx] = [float(row[k]) for k in ('x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w')]
            n_tracked += 1

    if n_dropped:
        log.warning(f'  {n_dropped} trajectory row(s) past the {n}-frame grid; ignored.')
    log.info(f'  Trajectory: {n_tracked} tracked, {n_lost} lost of {n_tracked + n_lost} fed frames')
    return poses


def _post_chirp_start(ep_grp: zarr.Group, n_total: int) -> tuple[int, bool]:
    """
    First frame index at/after the sync chirp ends, and whether the marker was found.

    The idle prefix before the chirp is where the localizer is still relocalizing and is
    trimmed at export anyway, so the usability gate judges the post-chirp window — the same
    span that actually reaches the dataset. Step 1 writes the marker, and steps run in order,
    so it is available here; a store predating it falls back to the whole episode.
    """
    if 'annotations/time_sync' not in ep_grp or 'timestamps/gopro' not in ep_grp:
        return 0, False
    chirp_end_s = grp(ep_grp, 'annotations/time_sync').attrs.get('gopro_chirp_end_s')
    if chirp_end_s is None:
        return 0, False
    gopro_ts = np.asarray(arr(ep_grp, 'timestamps/gopro')[:], dtype=np.float64)
    return int(np.searchsorted(gopro_ts, float(chirp_end_s), side='left')), True


def _write_slam_results(
    ep_grp: zarr.Group,
    poses: np.ndarray,
    settings_path: pathlib.Path,
    atlas_path: pathlib.Path,
    frame_stride: int = 1,
) -> None:
    """
    Write SLAM poses and summary annotations back into ep_grp.

    ``gopro/slam_poses`` is the localizer's forward trajectory, which is what the rest of the
    pipeline consumes. Nothing is gap-filled here or downstream: a frame SLAM could not place
    stays NaN, and the exporter turns runs of NaN into episode boundaries.

    All the frame counts are over the frames SLAM was actually *fed*
    (``0, stride, 2*stride, ...``), never over every frame.  Under decimation the skipped
    frames have no pose by construction, so a whole-grid count reads ~1/stride even for a
    perfect run and any threshold applied to it would condemn the entire corpus.  At stride 1
    the two definitions coincide.  ``n_frames_lost`` is the one exception, kept on the whole
    grid for backward compatibility -- do not gate on it.
    """
    gopro_grp = ep_grp.require_group('gopro')
    # Delete first so a re-run can't leave a stale array behind claiming to describe the
    # current poses -- including the slam_poses_{forward,reverse} pair that pzarr v3 wrote.
    for name in ('slam_poses', 'slam_poses_forward', 'slam_poses_reverse'):
        if name in gopro_grp:
            del gopro_grp[name]
    gopro_grp.create_array('slam_poses', data=poses, compressor=_BLOSC)

    is_lost = np.isnan(poses[:, 0])
    n_total = int(len(is_lost))
    n_lost = int(is_lost.sum())
    # count transitions lost→tracked (each run of tracked frames after a gap)
    transitions = int(np.count_nonzero(np.diff(is_lost.astype(np.int8)) == -1))

    fed_idx = np.arange(0, n_total, frame_stride)
    n_fed = int(len(fed_idx))
    n_fed_tracked = int((~is_lost[fed_idx]).sum()) if n_fed else 0

    i0, chirp_gated = _post_chirp_start(ep_grp, n_total)
    fed_post = fed_idx[fed_idx >= i0]
    n_fed_post = int(len(fed_post))
    n_fed_post_lost = int(is_lost[fed_post].sum()) if n_fed_post else 0

    slam_grp = ep_grp.require_group('annotations').require_group('slam')
    # Same reasoning as deleting the stale arrays above: a v3 store re-run under v4 would
    # otherwise keep attrs describing a two-pass merge that no longer happened — including
    # reverse_pass: True, which reads as a claim about *these* poses.
    for stale in (
        'reverse_pass',
        'reverse_merged',
        'reverse_n_filled',
        'reverse_n_forward_only',
        'reverse_overlap_frames',
        'reverse_overlap_median_mm',
    ):
        slam_grp.attrs.pop(stale, None)
    slam_grp.attrs['n_frames_total'] = n_total
    slam_grp.attrs['n_frames_lost'] = n_lost
    slam_grp.attrs['frame_stride'] = int(frame_stride)
    slam_grp.attrs['n_frames_fed'] = n_fed
    slam_grp.attrs['n_frames_fed_tracked'] = n_fed_tracked
    slam_grp.attrs['n_frames_fed_post_chirp'] = n_fed_post
    slam_grp.attrs['n_frames_fed_lost_post_chirp'] = n_fed_post_lost
    #: False when no chirp marker was found, so the two attrs above cover the whole episode.
    slam_grp.attrs['chirp_gated'] = chirp_gated
    slam_grp.attrs['tracking_ratio'] = float(n_fed_tracked) / n_fed if n_fed > 0 else 0.0
    slam_grp.attrs['n_relocalization_events'] = transitions
    slam_grp.attrs['orb_slam3_settings_path'] = str(settings_path.resolve())
    slam_grp.attrs['atlas_path'] = str(atlas_path.resolve())

    log.info(
        f'  SLAM results: {n_fed_tracked}/{n_fed} fed frames tracked '
        f'({100.0 * slam_grp.attrs["tracking_ratio"]:.1f}%)'
        + (f' at stride {frame_stride}' if frame_stride != 1 else '')
        + f'; {n_lost}/{n_total} frames without a pose overall, '
        f'{transitions} relocalization events'
    )


@register_preprocessing_step(step_number=2, step_name='orb-slam3')
class OrbSlam3Step(PreprocessingStep):
    """
    Estimate per-frame GoPro poses from video + IMU via monocular-inertial ORB-SLAM3.

    Phase 1 (map building): exports the MAPPING episode as an mp4 video plus
    GoPro-style telemetry JSON, invokes the map-building binary, saves the
    atlas sidecar, and writes the mapping pass' own per-frame trajectory
    (from VIBA) back to the MAPPING episode using the same schema as Phase 2.

    Phase 2 (localization): for each EPISODE group, exports as mp4 + JSON,
    invokes the localization binary against the pre-built atlas, and writes
    ``gopro/slam_poses`` (N,7) float64 ``[x,y,z, qx,qy,qz,qw]`` back into the
    zarr store (NaN rows = lost frames), plus summary annotations under
    ``annotations/slam``.

    The ORB-SLAM3 binaries themselves consume the existing
    ``mono_inertial_gopro_vi`` interface (video file + GoPro telemetry JSON),
    so the heavy lifting around format conversion happens here in Python.
    Atlas save/load paths are injected into a per-invocation temp copy of
    the settings YAML.

    Lost frames have all-NaN poses; downstream consumers must check
    NaN before using a pose.

    Constructor arguments
    ---------------------
    orb_slam3_dir:
        Root directory of the ORB-SLAM3 installation.  Defaults to the
        ``external/ORB_SLAM3_PolyUMI`` git submodule.  Override via the
        ``ORB_SLAM3_DIR`` env var to point at an out-of-tree build.
        Expected layout::

            {orb_slam3_dir}/
            ├── {bin_subdir}/          # default Examples/Monocular-Inertial
            │   ├── {map_builder_bin}  # mono_inertial_gopro_vi_polyumi
            │   └── {localizer_bin}    # mono_inertial_gopro_vi_localize
            └── Vocabulary/
                └── ORBvoc.txt

    settings_yaml:
        Path to the camera/IMU settings YAML.  Defaults to the bundled
        ``ingest/config/gopro_hero12_slam.yaml`` template; that template
        contains placeholder values and must be calibrated before use.

    map_builder_bin:
        Binary name (relative to ``orb_slam3_dir/bin/``) for the map-building
        mode.

    localizer_bin:
        Binary name for the localization mode.

    timeout_s:
        Per-episode subprocess timeout in seconds.  None = no timeout.
    """

    def __init__(
        self,
        orb_slam3_dir: pathlib.Path | None = None,
        settings_yaml: pathlib.Path | None = None,
        # Named `*_polyumi` to disambiguate from the Cheng fork's stock
        # mono_inertial_gopro_vi binary, which has hardcoded viewer=true and no
        # trajectory-output flag and is therefore not a drop-in replacement.
        map_builder_bin: str = 'mono_inertial_gopro_vi_polyumi',
        localizer_bin: str = 'mono_inertial_gopro_vi_localize',
        bin_subdir: str | None = None,
        timeout_s: float | None = None,
        resolution_divisor: int | None = None,
        localization_frame_stride: int | None = None,
    ) -> None:
        """
        Initialize the ORB-SLAM3 step.

        Parameters
        ----------
        orb_slam3_dir:
            Root of the ORB-SLAM3 installation.
        settings_yaml:
            Camera/IMU settings YAML.  Defaults to the Hero 12 template.
        map_builder_bin:
            Binary filename under ``orb_slam3_dir/bin_subdir/`` for map building.
        localizer_bin:
            Binary filename under ``orb_slam3_dir/bin_subdir/`` for localization.
        bin_subdir:
            Subdirectory of ``orb_slam3_dir`` that contains the binaries.
            Defaults to ``Examples/Monocular-Inertial`` to match the in-repo
            ORB_SLAM3_PolyUMI build layout.
        timeout_s:
            Per-episode subprocess timeout; None = no timeout.
        resolution_divisor:
            Downsample factor applied to the camera settings for *both* passes.  Map
            building and localization must use the same value: ORB descriptors and the
            scale pyramid are resolution-dependent, so localizing against a
            resolution-mismatched atlas makes relocalization unreliable.  Read from
            ``config/slam.yaml`` unless overridden by ``POLYUMI_SLAM_RES_DIV``.
        localization_frame_stride:
            Feed every Nth frame to the *localizer* (2 ~= 30 fps from a 59.94 fps
            source).  Map building is deliberately left at full rate -- decimating it
            measured no benefit and 20 fps mapping failed outright.  Read from
            ``config/slam.yaml`` unless overridden by ``POLYUMI_SLAM_LOC_STRIDE``.

        Neither has an in-code default.  These two numbers decide what ORB-SLAM3 ever sees,
        and they are recorded per episode (``annotations/slam/frame_stride``) and enforced to
        be uniform across a DP export -- so a fallback quietly disagreeing with the
        checked-in config is exactly the failure that would split a corpus across two
        incompatible time bases.  ``config/slam.yaml`` is the single source of truth.

        Raises
        ------
        FileNotFoundError
            If ``config/slam.yaml`` is missing.
        KeyError
            If it omits either key.
        ValueError
            If either value is below 1.

        """
        if orb_slam3_dir is None:
            orb_slam3_dir = pathlib.Path(os.environ.get('ORB_SLAM3_DIR', str(_DEFAULT_ORB_SLAM3_DIR)))
        if bin_subdir is None:
            bin_subdir = os.environ.get('ORB_SLAM3_BIN_SUBDIR', 'Examples/Monocular-Inertial')
        self.orb_slam3_dir = pathlib.Path(orb_slam3_dir)
        self.settings_yaml = pathlib.Path(settings_yaml) if settings_yaml else _DEFAULT_SETTINGS_YAML
        self.map_builder_bin = self.orb_slam3_dir / bin_subdir / map_builder_bin
        self.localizer_bin = self.orb_slam3_dir / bin_subdir / localizer_bin
        self.timeout_s = timeout_s
        # Precedence: explicit argument > environment override > config/slam.yaml.  There is
        # no fourth tier: see the docstring for why a fallback is worse than a hard failure.
        # Loaded here rather than at import so a broken config fails when a step is actually
        # constructed, not when anything in the package is imported.
        if resolution_divisor is None:
            resolution_divisor = int(
                os.environ.get('POLYUMI_SLAM_RES_DIV', _require_slam_setting('resolution_divisor'))
            )
        if localization_frame_stride is None:
            localization_frame_stride = int(
                os.environ.get('POLYUMI_SLAM_LOC_STRIDE', _require_slam_setting('localization_frame_stride'))
            )
        if resolution_divisor < 1 or localization_frame_stride < 1:
            raise ValueError(
                f'resolution_divisor and localization_frame_stride must be >= 1, got '
                f'{resolution_divisor} and {localization_frame_stride}'
            )
        self.resolution_divisor = resolution_divisor
        self.localization_frame_stride = localization_frame_stride

    @property
    def _vocab_path(self) -> pathlib.Path:
        return self.orb_slam3_dir / 'Vocabulary' / 'ORBvoc.txt'

    def _validate_settings_yaml(self) -> None:
        if not _SLAM_MASK_PNG.exists():
            # Hard failure rather than an unmasked fallback: an unmasked run does not crash,
            # it quietly produces a map nothing can relocalize against, and you only find out
            # after re-running 60 episodes.
            raise FileNotFoundError(
                f'Gripper mask not found: {_SLAM_MASK_PNG}. SLAM needs the camera-rigid '
                f'hardware (fingers, ArUco tags, LEDs, mirrors) blanked out; without it both '
                f'two-view init and relocalization degrade badly. Draw one against a '
                f'temporal-median frame — white = discard, black = keep.'
            )
        if not self.settings_yaml.exists():
            raise FileNotFoundError(f'ORB-SLAM3 settings YAML not found: {self.settings_yaml}')
        value_lines = [ln for ln in self.settings_yaml.read_text().splitlines() if not ln.lstrip().startswith('#')]
        if any(_PLACEHOLDER_MARKER in ln for ln in value_lines):
            raise RuntimeError(
                f'Settings YAML at {self.settings_yaml} still contains uncalibrated placeholder '
                f'values (search for "{_PLACEHOLDER_MARKER}"). Fill in camera intrinsics, Tbc, '
                f'and IMU noise parameters from a calibration run before using this step.'
            )

    def _run_subprocess(
        self,
        cmd: list[str],
        stdout_log: pathlib.Path,
        stderr_log: pathlib.Path,
        label: str,
        cwd: pathlib.Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        log.info(f'  Running: {" ".join(cmd)}')
        with open(stdout_log, 'w') as fout, open(stderr_log, 'w') as ferr:
            result = subprocess.run(
                cmd,
                stdout=fout,
                stderr=ferr,
                timeout=self.timeout_s,
                cwd=cwd,
                env=env,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f'{label} exited with code {result.returncode}. stderr: {stderr_log}  stdout: {stdout_log}'
            )

    def _build_map(
        self,
        ep_grp: zarr.Group,
        atlas_path: pathlib.Path,
        log_dir: pathlib.Path,
        scene_zarr: pathlib.Path,
    ) -> None:
        gopro_mp4 = resolve_gopro_mp4(ep_grp, scene_zarr)
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix='polyumi_slam_map_'))
        try:
            video_path, json_path, frame_ts = _export_episode(ep_grp, tmp_dir, gopro_mp4)
            # Half resolution, but full frame rate: decimating the mapping pass
            # measured no benefit and 20 fps mapping failed to initialise a map at
            # all, so only the resolution is reduced here.
            settings_path = _make_temp_settings_yaml(
                self.settings_yaml,
                tmp_dir,
                save_atlas=atlas_path,
                res_div=self.resolution_divisor,
            )
            traj_out = log_dir / 'mapping_trajectory.csv'
            cmd = [
                str(self.map_builder_bin),
                str(self._vocab_path),
                str(settings_path),
                str(video_path),
                str(json_path),
                str(traj_out),
            ]
            self._run_subprocess(
                cmd,
                log_dir / 'mapping_slam.stdout',
                log_dir / 'mapping_slam.stderr',
                label='ORB-SLAM3 map builder',
                cwd=log_dir,
            )
            if not atlas_path.exists():
                raise RuntimeError(f'ORB-SLAM3 map builder completed but atlas not found at {atlas_path}')

            # Put the mapping trajectory back onto the mapping episode and persist it
            # alongside the localized episodes' poses. Same CSV format and same parser as
            # phase 2 — keeps the schema consistent across MAPPING and EPISODE sessions.
            # Map building is never decimated, so its stride is 1.
            if traj_out.exists():
                log.info(f'  Mapping trajectory saved to {traj_out}')
                poses = _parse_trajectory_csv(traj_out, frame_ts)
                _write_slam_results(ep_grp, poses, self.settings_yaml, atlas_path)

            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            log.error(f'Map building failed; temp dir preserved for debugging: {tmp_dir}')
            raise

    def _localize_episode(
        self,
        ep_grp: zarr.Group,
        episode_index: int,
        atlas_path: pathlib.Path,
        log_dir: pathlib.Path,
        scene_zarr: pathlib.Path,
    ) -> None:
        gopro_mp4 = resolve_gopro_mp4(ep_grp, scene_zarr)
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f'polyumi_slam_ep{episode_index}_'))
        try:
            video_path, json_path, frame_ts = _export_episode(ep_grp, tmp_dir, gopro_mp4)
            # Resolution must match the atlas the map builder produced; the frame
            # rate is additionally divided by the stride so ORB-SLAM3's
            # keyframe-insertion window matches the rate it's actually being fed.
            settings_path = _make_temp_settings_yaml(
                self.settings_yaml,
                tmp_dir,
                load_atlas=atlas_path,
                res_div=self.resolution_divisor,
                fps_div=self.localization_frame_stride,
            )
            traj_out = tmp_dir / 'trajectory.csv'
            # Forward pass only. The binary still accepts an optional 6th argument that makes
            # it run a second, temporally-reversed pass against the same atlas; we no longer
            # pass it. Recovering the lead-in frames that way meant merging two trajectories
            # and trusting the result, and the pipeline now prefers to drop or split around
            # frames SLAM could not place. See docs and git history if it needs to come back.
            cmd = [
                str(self.localizer_bin),
                str(self._vocab_path),
                str(settings_path),
                str(video_path),
                str(json_path),
                str(traj_out),
            ]
            stdout_log = log_dir / f'episode_{episode_index}_slam.stdout'
            self._run_subprocess(
                cmd,
                stdout_log,
                log_dir / f'episode_{episode_index}_slam.stderr',
                label=f'ORB-SLAM3 localizer (episode {episode_index})',
                cwd=log_dir,
                env={
                    **os.environ,
                    'POLYUMI_SLAM_FRAME_STRIDE': str(self.localization_frame_stride),
                },
            )
            if not traj_out.exists():
                raise RuntimeError(f'ORB-SLAM3 localizer completed but trajectory file not found: {traj_out}')

            poses = _parse_trajectory_csv(traj_out, frame_ts, frame_stride=self.localization_frame_stride)
            _write_slam_results(
                ep_grp,
                poses,
                self.settings_yaml,
                atlas_path,
                frame_stride=self.localization_frame_stride,
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            log.error(f'Localization failed; temp dir preserved: {tmp_dir}')
            raise

    def prepare_scene(self, scene: SceneContext) -> None:
        """
        Phase 1: build the ORB-SLAM3 atlas from the scene's MAPPING session.

        Expects one episode group with ``session_type`` set to ``'MAPPING'`` (written by
        ``build_pzarr``). A scene recorded without a mapping pass has none, and rather than
        refuse it the step maps from its first episode — that scene's own walk is the only
        thing on offer, and it is what the operator implicitly asked for by recording no
        mapping session.

        A failure here is fatal to the step by design: without an atlas there is nothing for
        any episode to localize against, so this is not an episode-shaped failure.
        """
        self._validate_settings_yaml()

        scene_dir = scene.scene_dir
        self.atlas_path = scene_dir / f'{scene_dir.name}.atlas.osa'
        self.log_dir = scene_dir / 'slam_logs'
        self.log_dir.mkdir(exist_ok=True)

        episodes = scene.episodes
        mapping = next((ep for ep in episodes if ep.is_mapping), None)

        if mapping is None:
            log.warning(
                'No episode with session_type=MAPPING in this scene; mapping from its first '
                'episode instead. Expected for a scene recorded without a mapping walk; if you '
                'did record one, that session did not make it into the store.'
            )
            mapping = episodes[0]
        self._mapping_key = mapping.key

        # The map is built from this one episode, so a mapping session that failed to build
        # has no video or timestamps to feed the binary. Say so plainly — without this the
        # step dies further down on a bare `KeyError: 'timestamps/gopro'`.
        mapping_failure = mapping.failure
        if mapping_failure is not None:
            raise RuntimeError(
                f'{mapping.key} ({mapping.session_dir or "unknown session"}) is the mapping session but was '
                f'flagged unusable in {mapping_failure.step}: {mapping_failure.error}. '
                f'Fix or re-fetch that session and re-run; there is nothing to build a map from.'
            )

        if len(episodes) == 1:
            log.warning(
                f'No EPISODE groups found in {scene.zarr_path} — only {mapping.key} '
                f'(session_type=MAPPING) is present. Map will be built but no '
                f'localization will run. Add episode sessions to localize.'
            )

        if self.atlas_path.exists() and scene.force:
            log.info(f'--force: removing existing atlas at {self.atlas_path}')
            self.atlas_path.unlink()
        if self.atlas_path.exists():
            log.info(f'Atlas already exists at {self.atlas_path}, skipping map building.')
            return

        log.info(f'Phase 1: building map from {mapping.key}...')
        t0 = time.monotonic()
        self._build_map(mapping.group, self.atlas_path, self.log_dir, scene.zarr_path)
        elapsed = time.monotonic() - t0
        log.info(f'Map built in {elapsed:.1f}s: {self.atlas_path}')

    def process_episode(self, scene: SceneContext, episode: Episode) -> None:
        """Phase 2: localize one episode against the atlas built in phase 1."""
        # The mapping session *is* the map, so there is nothing to localize it against — its
        # own trajectory was already reconciled onto it during the phase-1 build.
        if episode.key == self._mapping_key:
            return

        log.info(f'Phase 2: localizing {episode.key}...')
        t0 = time.monotonic()
        # episode.index (parsed from the key) rather than a loop counter, so log filenames and
        # tmp dirs line up with the episode_N group in scene.zarr even though MAPPING is skipped.
        self._localize_episode(episode.group, episode.index, self.atlas_path, self.log_dir, scene.zarr_path)
        log.info(f'Localized {episode.key} in {time.monotonic() - t0:.1f}s')

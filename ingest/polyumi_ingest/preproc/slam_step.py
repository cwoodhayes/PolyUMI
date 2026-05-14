"""ORB-SLAM3 Monocular-Inertial preprocessing step."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING

import cv2
import imagecodecs.numcodecs  # noqa: F401 — registers imagecodecs_jpegxl with numcodecs
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation

from polyumi_ingest.preproc.step_base import (
    PreprocessingStep,
    register_preprocessing_step,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

# Marker string used in the settings YAML to flag values that need calibration.
_PLACEHOLDER_MARKER = 'CALIBRATE_ME'

_DEFAULT_SETTINGS_YAML = pathlib.Path(__file__).parent.parent.parent / 'config' / 'gopro_hero12_slam.yaml'

# Maximum acceptable distance (as a fraction of a frame period) between a
# trajectory entry's timestamp and the nearest frame timestamp when
# reconciling the C++ output back onto our frame index.
_TRAJ_TOLERANCE_FRAC = 0.5


def _arr(grp: zarr.Group, path: str) -> zarr.Array:
    return grp[path]  # type: ignore[return-value]


def _quat_to_se3(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Build a 4×4 SE3 matrix from translation + unit quaternion (x,y,z,w)."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
    mat[:3, 3] = [tx, ty, tz]
    return mat


def _export_video_mp4(
    frames_arr: zarr.Array,
    video_path: pathlib.Path,
    fps: float,
) -> None:
    """
    Encode all frames in ``frames_arr`` to an mp4 at the given constant fps.

    Frames in the zarr are stored as JpegXL-decoded RGB uint8; cv2.VideoWriter
    expects BGR. Encoding via the ``mp4v`` fourcc on Linux opencv builds is
    lossy but feature-preserving for SLAM at typical GoPro resolutions.
    """
    n = frames_arr.shape[0]
    if n == 0:
        raise RuntimeError('No frames to export')
    first = frames_arr[0]
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open cv2.VideoWriter for {video_path}')
    try:
        for i in range(n):
            rgb = frames_arr[i]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


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
    accl_interp = np.column_stack([
        np.interp(gyro_ts, accl_ts, accl[:, j]) for j in range(3)
    ])

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
) -> tuple[pathlib.Path, pathlib.Path, np.ndarray]:
    """
    Export an episode's frames + IMU to mp4 + telemetry JSON in ``tmp_dir``.

    Returns (video_path, json_path, frame_ts) where ``frame_ts`` is the
    per-frame UTC timestamp array (needed downstream to reconcile the
    C++ trajectory output back onto the original frame indices).
    """
    gopro_ts = np.asarray(_arr(ep_grp, 'timestamps/gopro')[:], dtype=np.float64)
    if len(gopro_ts) < 2:
        raise RuntimeError(f'Episode has fewer than 2 frames ({len(gopro_ts)})')
    fps = 1.0 / float(np.median(np.diff(gopro_ts)))
    log.info(f'  Episode fps (median from frame timestamps): {fps:.3f} ({len(gopro_ts)} frames)')

    video_path = tmp_dir / 'video.mp4'
    _export_video_mp4(_arr(ep_grp, 'gopro/frames'), video_path, fps)
    log.info(f'  Exported {len(gopro_ts)} frames to {video_path}')

    gyro = np.asarray(_arr(ep_grp, 'gopro/gyro')[:], dtype=np.float64)
    gyro_ts = np.asarray(_arr(ep_grp, 'timestamps/gopro_gyro')[:], dtype=np.float64)
    accl = np.asarray(_arr(ep_grp, 'gopro/accl')[:], dtype=np.float64)
    accl_ts = np.asarray(_arr(ep_grp, 'timestamps/gopro_accl')[:], dtype=np.float64)

    json_path = tmp_dir / 'telemetry.json'
    _export_telemetry_json(gyro, gyro_ts, accl, accl_ts, float(gopro_ts[0]), json_path)
    log.info(f'  Exported {len(gyro_ts)} IMU samples to {json_path}')

    return video_path, json_path, gopro_ts


def _make_temp_settings_yaml(
    src: pathlib.Path,
    tmp_dir: pathlib.Path,
    save_atlas: pathlib.Path | None = None,
    load_atlas: pathlib.Path | None = None,
) -> pathlib.Path:
    """
    Copy ``src`` settings YAML to ``tmp_dir`` with atlas paths appended.

    ORB-SLAM3 reads atlas save/load paths from the YAML
    (``System.SaveAtlasToFile`` / ``System.LoadAtlasFromFile``); the binary
    has no CLI flag for them. We inject the right key here so the canonical
    config file stays untouched.
    """
    content = src.read_text()
    if not content.endswith('\n'):
        content += '\n'
    content += '\nSystem.Viewer: 0\n'
    if save_atlas is not None:
        content += f'\nSystem.SaveAtlasToFile: "{save_atlas}"\n'
    if load_atlas is not None:
        content += f'\nSystem.LoadAtlasFromFile: "{load_atlas}"\n'
    dst = tmp_dir / 'settings.yaml'
    dst.write_text(content)
    return dst


def _parse_and_reconcile_trajectory(
    traj_path: pathlib.Path,
    frame_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse an ORB-SLAM3 EuRoC-format trajectory and align it to ``frame_ts``.

    ``SaveTrajectoryEuRoC`` writes whitespace-separated rows::

        timestamp_ns tx ty tz qx qy qz qw

    Lost frames are silently omitted, so we map each entry to its nearest
    frame timestamp (within half a frame period) and mark every frame that
    received no match as is_lost=True.

    The trajectory timestamps are nanoseconds of *video time* (because the
    C++ binary computes tframe from ``cap.get(CAP_PROP_POS_MSEC)``), so we
    add ``frame_ts[0]`` to bring them back to UTC before matching.

    Returns ``(poses, is_lost)`` shaped (N,4,4) float32 and (N,) bool. Lost
    rows in ``poses`` are all-NaN.
    """
    n = len(frame_ts)
    poses = np.full((n, 4, 4), np.nan, dtype=np.float32)
    is_lost = np.ones(n, dtype=bool)

    if n < 2:
        return poses, is_lost

    t_ref = float(frame_ts[0])
    period = float(np.median(np.diff(frame_ts)))
    tolerance = _TRAJ_TOLERANCE_FRAC * period

    n_matched = 0
    n_skipped = 0
    with open(traj_path) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 8:
                log.warning(f'  Skipping malformed trajectory line {line_no}: {line!r}')
                continue
            t_ns = float(parts[0])
            tx, ty, tz, qx, qy, qz, qw = (float(p) for p in parts[1:])
            t_utc = t_ref + t_ns / 1e9

            idx_right = int(np.searchsorted(frame_ts, t_utc))
            candidates = []
            if idx_right > 0:
                candidates.append(idx_right - 1)
            if idx_right < n:
                candidates.append(idx_right)
            idx = min(candidates, key=lambda i: abs(frame_ts[i] - t_utc))

            if abs(frame_ts[idx] - t_utc) > tolerance:
                log.warning(
                    f'  Trajectory entry at video t={t_ns / 1e9:.6f}s has no '
                    f'matching frame within {tolerance * 1000:.1f}ms — skipping'
                )
                n_skipped += 1
                continue

            poses[idx] = _quat_to_se3(tx, ty, tz, qx, qy, qz, qw)
            is_lost[idx] = False
            n_matched += 1

    log.info(f'  Trajectory reconciliation: {n_matched} matched, {n_skipped} skipped')
    return poses, is_lost


def _write_slam_results(
    ep_grp: zarr.Group,
    poses: np.ndarray,
    is_lost: np.ndarray,
    settings_path: pathlib.Path,
    atlas_path: pathlib.Path,
) -> None:
    """Write SLAM poses and summary annotations back into ep_grp."""
    gopro_grp = ep_grp.require_group('gopro')

    if 'slam_poses' in gopro_grp:
        del gopro_grp['slam_poses']
    if 'slam_is_lost' in gopro_grp:
        del gopro_grp['slam_is_lost']

    gopro_grp.create_array('slam_poses', data=poses, compressor=_BLOSC)
    gopro_grp.create_array('slam_is_lost', data=is_lost, compressor=_BLOSC)

    n_total = int(len(is_lost))
    n_lost = int(is_lost.sum())
    # count transitions lost→tracked (each run of tracked frames after a gap)
    transitions = int(np.count_nonzero(np.diff(is_lost.astype(np.int8)) == -1))

    slam_grp = ep_grp.require_group('annotations').require_group('slam')
    slam_grp.attrs['n_frames_total'] = n_total
    slam_grp.attrs['n_frames_lost'] = n_lost
    slam_grp.attrs['tracking_ratio'] = float(n_total - n_lost) / n_total if n_total > 0 else 0.0
    slam_grp.attrs['n_relocalization_events'] = transitions
    slam_grp.attrs['orb_slam3_settings_path'] = str(settings_path.resolve())
    slam_grp.attrs['atlas_path'] = str(atlas_path.resolve())

    log.info(
        f'  SLAM results: {n_total} frames, {n_lost} lost '
        f'({100.0 * n_lost / n_total:.1f}%), {transitions} relocalization events'
    )


@register_preprocessing_step(step_number=2, step_name='orb-slam3')
class OrbSlam3Step(PreprocessingStep):
    """
    ORB-SLAM3 Monocular-Inertial SLAM preprocessing step.

    Phase 1 (map building): exports the MAPPING episode as an mp4 video plus
    GoPro-style telemetry JSON, invokes the map-building binary, and saves the
    atlas sidecar.

    Phase 2 (localization): for each EPISODE group, exports as mp4 + JSON,
    invokes the localization binary against the pre-built atlas, and writes
    ``gopro/slam_poses`` (N,4,4) float32 and ``gopro/slam_is_lost`` (N,) bool
    back into the zarr store, plus summary annotations under
    ``annotations/slam``.

    The ORB-SLAM3 binaries themselves consume the existing
    ``mono_inertial_gopro_vi`` interface (video file + GoPro telemetry JSON),
    so the heavy lifting around format conversion happens here in Python.
    Atlas save/load paths are injected into a per-invocation temp copy of
    the settings YAML.

    Lost frames have all-NaN poses; downstream consumers must check
    ``slam_is_lost`` or test for NaN before using a pose.

    Constructor arguments
    ---------------------
    orb_slam3_dir:
        Root directory of the ORB-SLAM3 installation.  Expected layout::

            {orb_slam3_dir}/
            ├── {bin_subdir}/          # default "bin"; source build uses
            │   ├── {map_builder_bin}  # "Examples/Monocular-Inertial"
            │   └── {localizer_bin}
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
        orb_slam3_dir: pathlib.Path = pathlib.Path(
            os.environ.get('ORB_SLAM3_DIR', '/usr/local/lib/ORB_SLAM3')
        ),
        settings_yaml: pathlib.Path | None = None,
        map_builder_bin: str = 'mono_inertial_gopro_vi',
        localizer_bin: str = 'mono_inertial_gopro_vi_localize',
        bin_subdir: str = os.environ.get('ORB_SLAM3_BIN_SUBDIR', 'bin'),
        timeout_s: float | None = None,
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
            Defaults to ``bin``; use ``Examples/Monocular-Inertial`` for a
            standard ORB-SLAM3 source build.
        timeout_s:
            Per-episode subprocess timeout; None = no timeout.

        """
        self.orb_slam3_dir = pathlib.Path(orb_slam3_dir)
        self.settings_yaml = pathlib.Path(settings_yaml) if settings_yaml else _DEFAULT_SETTINGS_YAML
        self.map_builder_bin = self.orb_slam3_dir / bin_subdir / map_builder_bin
        self.localizer_bin = self.orb_slam3_dir / bin_subdir / localizer_bin
        self.timeout_s = timeout_s

    @property
    def _vocab_path(self) -> pathlib.Path:
        return self.orb_slam3_dir / 'Vocabulary' / 'ORBvoc.txt'

    def _validate_settings_yaml(self) -> None:
        if not self.settings_yaml.exists():
            raise FileNotFoundError(f'ORB-SLAM3 settings YAML not found: {self.settings_yaml}')
        value_lines = [
            ln for ln in self.settings_yaml.read_text().splitlines()
            if not ln.lstrip().startswith('#')
        ]
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
    ) -> None:
        log.info(f'  Running: {" ".join(cmd)}')
        with open(stdout_log, 'w') as fout, open(stderr_log, 'w') as ferr:
            result = subprocess.run(
                cmd,
                stdout=fout,
                stderr=ferr,
                timeout=self.timeout_s,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f'{label} exited with code {result.returncode}. '
                f'stderr: {stderr_log}  stdout: {stdout_log}'
            )

    def _build_map(
        self,
        ep_grp: zarr.Group,
        atlas_path: pathlib.Path,
        log_dir: pathlib.Path,
    ) -> None:
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix='polyumi_slam_map_'))
        try:
            video_path, json_path, _ = _export_episode(ep_grp, tmp_dir)
            settings_path = _make_temp_settings_yaml(
                self.settings_yaml, tmp_dir, save_atlas=atlas_path,
            )
            cmd = [
                str(self.map_builder_bin),
                str(self._vocab_path),
                str(settings_path),
                str(video_path),
                str(json_path),
            ]
            self._run_subprocess(
                cmd,
                log_dir / 'mapping_slam.stdout',
                log_dir / 'mapping_slam.stderr',
                label='ORB-SLAM3 map builder',
            )
            if not atlas_path.exists():
                raise RuntimeError(
                    f'ORB-SLAM3 map builder completed but atlas not found at {atlas_path}'
                )
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
    ) -> None:
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f'polyumi_slam_ep{episode_index}_'))
        try:
            video_path, json_path, frame_ts = _export_episode(ep_grp, tmp_dir)
            settings_path = _make_temp_settings_yaml(
                self.settings_yaml, tmp_dir, load_atlas=atlas_path,
            )
            traj_out = tmp_dir / 'trajectory.txt'
            cmd = [
                str(self.localizer_bin),
                str(self._vocab_path),
                str(settings_path),
                str(video_path),
                str(json_path),
                str(traj_out),
            ]
            self._run_subprocess(
                cmd,
                log_dir / f'episode_{episode_index}_slam.stdout',
                log_dir / f'episode_{episode_index}_slam.stderr',
                label=f'ORB-SLAM3 localizer (episode {episode_index})',
            )
            if not traj_out.exists():
                raise RuntimeError(
                    f'ORB-SLAM3 localizer completed but trajectory file not found: {traj_out}'
                )

            poses, is_lost = _parse_and_reconcile_trajectory(traj_out, frame_ts)
            _write_slam_results(ep_grp, poses, is_lost, self.settings_yaml, atlas_path)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            log.error(f'Localization failed; temp dir preserved: {tmp_dir}')
            raise

    def run_step(self, scene_zarr: pathlib.Path) -> None:
        """
        Run map building then per-episode localization on scene_zarr.

        Expects at least one episode group with ``session_type`` attribute set
        to ``'MAPPING'`` (written by ``build_pzarr``).  Falls back to treating
        the first episode as the mapping session for zarr stores built before
        this change was introduced (see OQ-4).
        """
        self._validate_settings_yaml()

        scene_dir = scene_zarr.parent
        atlas_path = scene_dir / f'{scene_dir.name}.atlas.osa'
        log_dir = scene_dir / 'slam_logs'
        log_dir.mkdir(exist_ok=True)

        root = zarr.open_group(str(scene_zarr), mode='a')
        episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
        if not episodes:
            raise RuntimeError(f'No episodes found in {scene_zarr}')

        mapping_key: str | None = None
        episode_keys: list[str] = []
        for ep_key in episodes:
            ep = root[ep_key]
            session_type = ep.attrs.get('session_type', None)
            if session_type == 'MAPPING':
                mapping_key = ep_key
            else:
                episode_keys.append(ep_key)

        # OQ-4 fallback: if no episode has session_type='MAPPING', treat first as mapping
        if mapping_key is None:
            log.warning(
                'No episode with session_type=MAPPING found; treating first episode as mapping. '
                'Rebuild the zarr store to get proper session_type attributes.'
            )
            mapping_key = episodes[0]
            episode_keys = [k for k in episodes if k != mapping_key]

        if not episode_keys:
            log.warning(
                f'No EPISODE groups found in {scene_zarr} — only {mapping_key} '
                f'(session_type=MAPPING) is present. Map will be built but no '
                f'localization will run. Add episode sessions to localize.'
            )

        # Phase 1: map building
        if atlas_path.exists():
            log.info(f'Atlas already exists at {atlas_path}, skipping map building.')
        else:
            log.info(f'Phase 1: building map from {mapping_key}...')
            self._build_map(root[mapping_key], atlas_path, log_dir)
            log.info(f'Map built: {atlas_path}')

        # Phase 2: per-episode localization
        for i, ep_key in enumerate(episode_keys):
            log.info(f'Phase 2: localizing {ep_key} ({i + 1}/{len(episode_keys)})...')
            self._localize_episode(root[ep_key], i, atlas_path, log_dir)

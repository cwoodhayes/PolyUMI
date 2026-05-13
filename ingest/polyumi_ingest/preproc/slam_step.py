"""ORB-SLAM3 Monocular-Inertial preprocessing step."""

from __future__ import annotations

import csv
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import cv2
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


def _arr(grp: zarr.Group, path: str) -> zarr.Array:
    return grp[path]  # type: ignore[return-value]


def _quat_to_se3(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Build a 4×4 SE3 matrix from translation + unit quaternion (x,y,z,w)."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
    mat[:3, 3] = [tx, ty, tz]
    return mat


def _export_frames(
    frames_arr: zarr.Array,
    ts: np.ndarray,
    frames_dir: pathlib.Path,
) -> None:
    """
    Decode JpegXL frames from zarr and write individual JPEGs to frames_dir.

    Files are named by UTC timestamp in seconds: ``{ts:.6f}.jpg``.
    Parallel decode+write via ThreadPoolExecutor mirrors write_frames_to_zarr.
    """
    n_workers = min(os.cpu_count() or 1, len(ts))

    def _write(i: int) -> None:
        frame = frames_arr[i]  # (H, W, 3) uint8 RGB, JpegXL decompressed by zarr
        frame_path = frames_dir / f'{float(ts[i]):.6f}.jpg'
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), bgr)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_write, i) for i in range(len(ts))]
    for fut in futures:
        fut.result()


def _export_imu_csv(
    gyro: np.ndarray,
    gyro_ts: np.ndarray,
    accl: np.ndarray,
    accl_ts: np.ndarray,
    csv_path: pathlib.Path,
) -> None:
    """
    Write IMU data to a CSV with columns: timestamp,gx,gy,gz,ax,ay,az.

    GoPro stores IMU in (z,x,y) axis order; this reorders to (x,y,z) to match
    what the ORB-SLAM3 binary and Tbc calibration expect (same reorder as
    mono_inertial_gopro_vi.cc: value[1],value[2],value[0]).

    Accelerometer samples are interpolated onto the gyroscope timestamps since
    the two sensors may run at slightly different rates.
    """
    # reorder axes: GoPro [z,x,y] → body [x,y,z]
    gyro_xyz = gyro[:, [1, 2, 0]]
    accl_xyz_full = accl[:, [1, 2, 0]]

    # interpolate accelerometer onto gyro timestamps
    accl_interp = np.column_stack([
        np.interp(gyro_ts, accl_ts, accl_xyz_full[:, j]) for j in range(3)
    ])

    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
        for i in range(len(gyro_ts)):
            writer.writerow([
                f'{gyro_ts[i]:.9f}',
                *[f'{v:.9f}' for v in gyro_xyz[i]],
                *[f'{v:.9f}' for v in accl_interp[i]],
            ])


def _export_episode(ep_grp: zarr.Group, tmp_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Export frames and IMU from an episode group to tmp_dir.

    Returns (frames_dir, imu_csv_path).
    """
    frames_dir = tmp_dir / 'frames'
    frames_dir.mkdir()

    gopro_ts = np.asarray(_arr(ep_grp, 'timestamps/gopro')[:], dtype=np.float64)
    frames_arr = _arr(ep_grp, 'gopro/frames')
    _export_frames(frames_arr, gopro_ts, frames_dir)
    log.info(f'  Exported {len(gopro_ts)} frames to {frames_dir}')

    gyro = np.asarray(_arr(ep_grp, 'gopro/gyro')[:], dtype=np.float64)
    gyro_ts = np.asarray(_arr(ep_grp, 'timestamps/gopro_gyro')[:], dtype=np.float64)
    accl = np.asarray(_arr(ep_grp, 'gopro/accl')[:], dtype=np.float64)
    accl_ts = np.asarray(_arr(ep_grp, 'timestamps/gopro_accl')[:], dtype=np.float64)

    imu_csv = tmp_dir / 'imu.csv'
    _export_imu_csv(gyro, gyro_ts, accl, accl_ts, imu_csv)
    log.info(f'  Exported {len(gyro_ts)} IMU samples to {imu_csv}')

    return frames_dir, imu_csv


def _parse_trajectory_csv(csv_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse an ORB-SLAM3 trajectory CSV.

    Expected columns: timestamp,tx,ty,tz,qx,qy,qz,qw,is_lost

    Returns (timestamps, poses, is_lost) where:
      - timestamps: (N,) float64
      - poses: (N,4,4) float32 SE3 matrices; all-NaN for lost frames
      - is_lost: (N,) bool
    """
    rows = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    n = len(rows)
    timestamps = np.zeros(n, dtype=np.float64)
    poses = np.zeros((n, 4, 4), dtype=np.float32)
    is_lost = np.zeros(n, dtype=bool)

    for i, row in enumerate(rows):
        timestamps[i] = float(row['timestamp'])
        lost = bool(int(row['is_lost']))
        is_lost[i] = lost
        if lost:
            poses[i] = np.full((4, 4), np.nan, dtype=np.float32)
        else:
            poses[i] = _quat_to_se3(
                float(row['tx']), float(row['ty']), float(row['tz']),
                float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw']),
            )

    return timestamps, poses, is_lost


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

    Phase 1 (map building): exports the MAPPING episode's frames and IMU to a
    temp directory, invokes the map-building binary, and saves the atlas sidecar.

    Phase 2 (localization): for each EPISODE group, exports frames and IMU,
    invokes the localization binary against the pre-built atlas, and writes
    ``gopro/slam_poses`` (N,4,4) float32 and ``gopro/slam_is_lost`` (N,) bool
    back into the zarr store, plus summary annotations under
    ``annotations/slam``.

    Lost frames have all-NaN poses; downstream consumers must check
    ``slam_is_lost`` or test for NaN before using a pose.

    Constructor arguments
    ---------------------
    orb_slam3_dir:
        Root directory of the ORB-SLAM3 installation.  Expected layout::

            {orb_slam3_dir}/
            ├── bin/
            │   ├── {map_builder_bin}
            │   └── {localizer_bin}
            └── Vocabulary/
                └── ORBvoc.txt

    settings_yaml:
        Path to the camera/IMU settings YAML.  Defaults to the bundled
        ``ingest/config/gopro_hero12_slam.yaml`` template; that template
        contains placeholder values and must be calibrated before use.

    map_builder_bin:
        Binary name (relative to ``orb_slam3_dir/bin/``) for the map-building
        mode.  See OQ-2 above.

    localizer_bin:
        Binary name for the localization mode.  See OQ-2 above.

    timeout_s:
        Per-episode subprocess timeout in seconds.  None = no timeout.
    """

    def __init__(
        self,
        orb_slam3_dir: pathlib.Path = pathlib.Path('/usr/local/lib/ORB_SLAM3'),
        settings_yaml: pathlib.Path | None = None,
        map_builder_bin: str = 'mono_inertial_gopro_vi',
        localizer_bin: str = 'mono_inertial_gopro_vi_localize',
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
            Binary filename under orb_slam3_dir/bin/ for map building.
        localizer_bin:
            Binary filename under orb_slam3_dir/bin/ for localization.
        timeout_s:
            Per-episode subprocess timeout; None = no timeout.

        """
        self.orb_slam3_dir = pathlib.Path(orb_slam3_dir)
        self.settings_yaml = pathlib.Path(settings_yaml) if settings_yaml else _DEFAULT_SETTINGS_YAML
        self.map_builder_bin = self.orb_slam3_dir / 'bin' / map_builder_bin
        self.localizer_bin = self.orb_slam3_dir / 'bin' / localizer_bin
        self.timeout_s = timeout_s

    @property
    def _vocab_path(self) -> pathlib.Path:
        return self.orb_slam3_dir / 'Vocabulary' / 'ORBvoc.txt'

    def _validate_settings_yaml(self) -> None:
        if not self.settings_yaml.exists():
            raise FileNotFoundError(f'ORB-SLAM3 settings YAML not found: {self.settings_yaml}')
        content = self.settings_yaml.read_text()
        if _PLACEHOLDER_MARKER in content:
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
            frames_dir, imu_csv = _export_episode(ep_grp, tmp_dir)
            cmd = [
                str(self.map_builder_bin),
                str(self._vocab_path),
                str(self.settings_yaml),
                str(frames_dir),
                str(imu_csv),
                str(atlas_path),
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
            frames_dir, imu_csv = _export_episode(ep_grp, tmp_dir)
            traj_csv = tmp_dir / 'trajectory.csv'
            cmd = [
                str(self.localizer_bin),
                str(self._vocab_path),
                str(self.settings_yaml),
                str(frames_dir),
                str(imu_csv),
                str(atlas_path),
                str(traj_csv),
            ]
            self._run_subprocess(
                cmd,
                log_dir / f'episode_{episode_index}_slam.stdout',
                log_dir / f'episode_{episode_index}_slam.stderr',
                label=f'ORB-SLAM3 localizer (episode {episode_index})',
            )
            if not traj_csv.exists():
                raise RuntimeError(
                    f'ORB-SLAM3 localizer completed but trajectory CSV not found: {traj_csv}'
                )

            _, poses, is_lost = _parse_trajectory_csv(traj_csv)
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
            raise RuntimeError(
                f'No EPISODE groups found to localize in {scene_zarr} '
                f'(only mapping episode {mapping_key} present).'
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

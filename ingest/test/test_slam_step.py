"""Tests for the ORB-SLAM3 SLAM preprocessing step."""

from __future__ import annotations

import csv
import os
import pathlib
import unittest.mock as mock

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from polyumi_ingest.preproc.slam_step import (
    OrbSlam3Step,
    _export_imu_csv,
    _parse_trajectory_csv,
    _quat_to_se3,
)

_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)


# ---------------------------------------------------------------------------
# Helpers for building minimal test zarr stores
# ---------------------------------------------------------------------------


def _make_episode(
    root: zarr.Group,
    key: str,
    n_frames: int = 4,
    n_imu: int = 20,
    session_type: str = 'EPISODE',
) -> zarr.Group:
    ep = root.require_group(key)
    ep.attrs['session_type'] = session_type

    H, W = 4, 6
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (n_frames, H, W, 3), dtype=np.uint8)
    gopro_ts = 1_000.0 + np.arange(n_frames, dtype=np.float64) / 60.0

    gopro_grp = ep.require_group('gopro')
    gopro_grp.zeros('frames', shape=(n_frames, H, W, 3), dtype='uint8', chunks=(1, H, W, 3))
    gopro_grp['frames'][:] = frames
    gopro_grp.create_array('gyro', data=rng.standard_normal((n_imu, 3)).astype(np.float32))
    gopro_grp.create_array('accl', data=rng.standard_normal((n_imu, 3)).astype(np.float32))

    ts_grp = ep.require_group('timestamps')
    ts_grp.create_array('gopro', data=gopro_ts)
    ts_grp.create_array('gopro_gyro', data=1_000.0 + np.arange(n_imu, dtype=np.float64) / 200.0)
    ts_grp.create_array('gopro_accl', data=1_000.0 + np.arange(n_imu, dtype=np.float64) / 200.0)

    ep.require_group('annotations')
    return ep


def _make_traj_csv(path: pathlib.Path, n: int, n_lost: int = 0) -> None:
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw', 'is_lost'])
        for i in range(n):
            lost = 1 if i < n_lost else 0
            writer.writerow([f'{1000.0 + i / 60.0:.9f}', 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, lost])


def _calibrated_settings(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a minimal settings YAML with no placeholder markers."""
    yaml_path = tmp_path / 'test_slam.yaml'
    yaml_path.write_text('%YAML:1.0\nCamera.fx: 200.0\nCamera.fy: 200.0\n')
    return yaml_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_mapping_episode_skipped_during_localization(tmp_path: pathlib.Path) -> None:
    """
    The MAPPING episode must be used only for map building, not localized.

    Verifies that run_step calls the map builder exactly once for episode_0
    (MAPPING) and the localizer for episode_1 (EPISODE), and does NOT call
    the localizer for episode_0.
    """
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    _make_episode(root, 'episode_0', session_type='MAPPING')
    _make_episode(root, 'episode_1', session_type='EPISODE')

    settings = _calibrated_settings(tmp_path)
    step = OrbSlam3Step(settings_yaml=settings)

    called_build = []
    called_localize = []

    def _fake_build(ep_grp, atlas_path, log_dir):
        called_build.append(ep_grp.name)
        atlas_path.touch()

    def _fake_localize(ep_grp, episode_index, atlas_path, log_dir):
        called_localize.append(ep_grp.name)
        n_frames = len(ep_grp['timestamps/gopro'])
        poses = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
        is_lost = np.zeros(n_frames, dtype=bool)
        from polyumi_ingest.preproc.slam_step import _write_slam_results
        _write_slam_results(ep_grp, poses, is_lost, settings, atlas_path)

    with mock.patch.object(step, '_build_map', side_effect=_fake_build), \
         mock.patch.object(step, '_localize_episode', side_effect=_fake_localize):
        step.run_step(scene_zarr)

    assert len(called_build) == 1
    assert called_build[0] == '/episode_0'
    assert len(called_localize) == 1
    assert called_localize[0] == '/episode_1'


def test_zarr_output_schema(tmp_path: pathlib.Path) -> None:
    """Verify that a mocked localization writes the correct zarr arrays and annotation attributes."""
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    _make_episode(root, 'episode_0', session_type='MAPPING', n_frames=6)
    _make_episode(root, 'episode_1', session_type='EPISODE', n_frames=6)

    n_frames = 6
    settings = _calibrated_settings(tmp_path)
    step = OrbSlam3Step(settings_yaml=settings)

    def _fake_build(ep_grp, atlas_path, log_dir):
        atlas_path.touch()

    def _fake_localize(ep_grp, episode_index, atlas_path, log_dir):
        traj_csv = tmp_path / f'traj_{episode_index}.csv'
        _make_traj_csv(traj_csv, n=n_frames, n_lost=2)
        _, poses, is_lost = _parse_trajectory_csv(traj_csv)
        from polyumi_ingest.preproc.slam_step import _write_slam_results
        _write_slam_results(ep_grp, poses, is_lost, settings, atlas_path)

    with mock.patch.object(step, '_build_map', side_effect=_fake_build), \
         mock.patch.object(step, '_localize_episode', side_effect=_fake_localize):
        step.run_step(scene_zarr)

    ep1 = zarr.open_group(str(scene_zarr / 'episode_1'), mode='r')

    # Array names and shapes
    assert 'gopro/slam_poses' in ep1
    assert 'gopro/slam_is_lost' in ep1
    assert ep1['gopro/slam_poses'].shape == (n_frames, 4, 4)
    assert ep1['gopro/slam_poses'].dtype == np.float32
    assert ep1['gopro/slam_is_lost'].shape == (n_frames,)
    assert ep1['gopro/slam_is_lost'].dtype == bool

    # Annotation attribute keys
    slam_attrs = ep1['annotations/slam'].attrs
    for key in ('n_frames_total', 'n_frames_lost', 'tracking_ratio',
                'n_relocalization_events', 'orb_slam3_settings_path', 'atlas_path'):
        assert key in slam_attrs, f'Missing annotation key: {key}'

    assert int(slam_attrs['n_frames_total']) == n_frames
    assert int(slam_attrs['n_frames_lost']) == 2
    assert abs(float(slam_attrs['tracking_ratio']) - (4 / 6)) < 1e-5

    # Lost frames → all-NaN pose
    poses_arr = ep1['gopro/slam_poses'][:]
    for i in range(2):  # first 2 rows were lost
        assert np.all(np.isnan(poses_arr[i]))


def test_placeholder_detection_raises(tmp_path: pathlib.Path) -> None:
    """Settings YAML with placeholder values must raise before any subprocess is called."""
    yaml_with_placeholder = tmp_path / 'bad.yaml'
    yaml_with_placeholder.write_text(
        '%YAML:1.0\nCamera.fx: 0.0  # CALIBRATE_ME\n'
    )
    step = OrbSlam3Step(settings_yaml=yaml_with_placeholder)
    with pytest.raises(RuntimeError, match='CALIBRATE_ME'):
        step.run_step(tmp_path / 'scene.zarr')


def test_imu_csv_axis_reorder(tmp_path: pathlib.Path) -> None:
    """GoPro (z,x,y) IMU axes are reordered to (x,y,z) in the exported CSV."""
    n = 10
    gyro = np.zeros((n, 3), dtype=np.float64)
    gyro[:, 0] = 1.0  # z-axis
    gyro[:, 1] = 2.0  # x-axis
    gyro[:, 2] = 3.0  # y-axis
    gyro_ts = np.arange(n, dtype=np.float64) / 200.0
    accl = np.zeros_like(gyro)
    accl_ts = gyro_ts.copy()

    csv_path = tmp_path / 'imu.csv'
    _export_imu_csv(gyro, gyro_ts, accl, accl_ts, csv_path)

    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        row = next(reader)

    # expected: gx=2.0 (orig col 1), gy=3.0 (orig col 2), gz=1.0 (orig col 0)
    assert abs(float(row['gx']) - 2.0) < 1e-6
    assert abs(float(row['gy']) - 3.0) < 1e-6
    assert abs(float(row['gz']) - 1.0) < 1e-6


def test_quat_to_se3_identity() -> None:
    """Identity quaternion should yield identity SE3."""
    mat = _quat_to_se3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    np.testing.assert_array_almost_equal(mat, np.eye(4))


def test_parse_trajectory_csv_roundtrip(tmp_path: pathlib.Path) -> None:
    """Trajectory CSV parse: lost frames get identity, tracked frames get real SE3."""
    traj_csv = tmp_path / 'traj.csv'
    _make_traj_csv(traj_csv, n=4, n_lost=1)
    ts, poses, is_lost = _parse_trajectory_csv(traj_csv)

    assert len(ts) == 4
    assert is_lost[0]
    assert not is_lost[1]
    assert np.all(np.isnan(poses[0]))  # lost → all-NaN
    # tracked: quaternion (0,0,0,1) → identity rotation, translation (0.1,0.2,0.3)
    np.testing.assert_array_almost_equal(poses[1, :3, :3], np.eye(3), decimal=5)
    np.testing.assert_array_almost_equal(poses[1, :3, 3], [0.1, 0.2, 0.3], decimal=5)


# ---------------------------------------------------------------------------
# Smoke test (skipped unless POLYUMI_TEST_SCENE_DIR is set)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get('POLYUMI_TEST_SCENE_DIR'),
    reason='Set POLYUMI_TEST_SCENE_DIR to a real scene directory to run this test',
)
def test_slam_step_smoke() -> None:
    """Full end-to-end run of OrbSlam3Step on a real scene directory."""
    scene_dir = pathlib.Path(os.environ['POLYUMI_TEST_SCENE_DIR'])
    step = OrbSlam3Step()
    step.run(scene_dir)

    scene_zarr = scene_dir / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='r')
    episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
    episode_keys = [k for k in episodes if root[k].attrs.get('session_type') != 'MAPPING']
    assert episode_keys, 'No episode groups found after SLAM step'
    for ep_key in episode_keys:
        ep = root[ep_key]
        assert 'gopro/slam_poses' in ep
        assert 'gopro/slam_is_lost' in ep
        assert 'annotations/slam' in ep

"""Tests for the ORB-SLAM3 SLAM preprocessing step."""

from __future__ import annotations

import json
import os
import pathlib
import unittest.mock as mock

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from polyumi_ingest.preproc.slam_step import (
    OrbSlam3Step,
    _export_telemetry_json,
    _make_temp_settings_yaml,
    _parse_and_reconcile_trajectory,
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
    gopro_grp.create_array('frames', data=frames, chunks=(1, H, W, 3))
    gopro_grp.create_array('gyro', data=rng.standard_normal((n_imu, 3)).astype(np.float32))
    gopro_grp.create_array('accl', data=rng.standard_normal((n_imu, 3)).astype(np.float32))

    ts_grp = ep.require_group('timestamps')
    ts_grp.create_array('gopro', data=gopro_ts)
    ts_grp.create_array('gopro_gyro', data=1_000.0 + np.arange(n_imu, dtype=np.float64) / 200.0)
    ts_grp.create_array('gopro_accl', data=1_000.0 + np.arange(n_imu, dtype=np.float64) / 200.0)

    ep.require_group('annotations')
    return ep


def _make_euroc_trajectory(
    path: pathlib.Path,
    frame_ts: np.ndarray,
    tracked_mask: np.ndarray,
) -> None:
    """
    Write a fake EuRoC-format trajectory output for frames where ``tracked_mask`` is True.

    Each row is whitespace-separated::

        timestamp_ns tx ty tz qx qy qz qw

    Timestamps are nanoseconds of *video time* (relative to frame_ts[0]),
    matching what ORB-SLAM3's SaveTrajectoryEuRoC writes for the gopro
    binary's tframe values.
    """
    t_ref = float(frame_ts[0])
    with open(path, 'w') as fh:
        for i, ts in enumerate(frame_ts):
            if not tracked_mask[i]:
                continue
            t_ns = (float(ts) - t_ref) * 1e9
            fh.write(f'{t_ns:.6f} 0.1 0.2 0.3 0.0 0.0 0.0 1.0\n')


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

    def _fake_build(ep_grp, atlas_path, log_dir, gopro_mp4=None):
        called_build.append(ep_grp.name)
        atlas_path.touch()

    def _fake_localize(ep_grp, episode_index, atlas_path, log_dir, gopro_mp4=None):
        called_localize.append(ep_grp.name)
        n_frames = ep_grp['timestamps/gopro'].shape[0]
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

    def _fake_build(ep_grp, atlas_path, log_dir, gopro_mp4=None):
        atlas_path.touch()

    def _fake_localize(ep_grp, episode_index, atlas_path, log_dir, gopro_mp4=None):
        traj_path = tmp_path / f'traj_{episode_index}.txt'
        frame_ts = np.asarray(ep_grp['timestamps/gopro'][:], dtype=np.float64)
        tracked = np.ones(n_frames, dtype=bool)
        tracked[:2] = False  # first two frames lost
        _make_euroc_trajectory(traj_path, frame_ts, tracked)
        poses, is_lost = _parse_and_reconcile_trajectory(traj_path, frame_ts)
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


def test_telemetry_json_preserves_raw_gopro_axis_order(tmp_path: pathlib.Path) -> None:
    """
    The exported telemetry JSON must preserve raw GoPro [z,x,y] axis order.

    The mono_inertial_gopro_vi binary reorders axes itself via
    ``value[1], value[2], value[0]`` → body [x,y,z]; if we reorder on the
    Python side too we'd double-rotate the IMU.
    """
    n = 10
    gyro = np.zeros((n, 3), dtype=np.float64)
    gyro[:, 0] = 1.0  # GoPro z-axis
    gyro[:, 1] = 2.0  # GoPro x-axis
    gyro[:, 2] = 3.0  # GoPro y-axis
    gyro_ts = 1000.0 + np.arange(n, dtype=np.float64) / 200.0
    accl = gyro.copy()
    accl_ts = gyro_ts.copy()

    json_path = tmp_path / 'telemetry.json'
    _export_telemetry_json(gyro, gyro_ts, accl, accl_ts, t_ref=1000.0, json_path=json_path)

    with open(json_path) as fh:
        blob = json.load(fh)

    gyro_samples = blob['1']['streams']['GYRO']['samples']
    assert len(gyro_samples) == n

    # First sample should carry raw [z=1.0, x=2.0, y=3.0]
    val0 = gyro_samples[0]['value']
    assert abs(val0[0] - 1.0) < 1e-6
    assert abs(val0[1] - 2.0) < 1e-6
    assert abs(val0[2] - 3.0) < 1e-6

    # cts is ms relative to t_ref → first sample at 0
    assert abs(float(gyro_samples[0]['cts']) - 0.0) < 1e-6
    # 200 Hz sampling → 5 ms between samples
    assert abs(float(gyro_samples[1]['cts']) - 5.0) < 1e-6


def test_make_temp_settings_yaml_injects_atlas_paths(tmp_path: pathlib.Path) -> None:
    """The temp YAML must contain the requested atlas key without losing the source content."""
    src = tmp_path / 'src.yaml'
    src.write_text('%YAML:1.0\nCamera.fx: 200.0\n')

    save_dst = _make_temp_settings_yaml(src, tmp_path, save_atlas=tmp_path / 'a.osa')
    content = save_dst.read_text()
    assert 'Camera.fx: 200.0' in content
    assert f'System.SaveAtlasToFile: "{tmp_path / "a.osa"}"' in content
    assert 'System.LoadAtlasFromFile' not in content

    load_dir = tmp_path / 'subdir'
    load_dir.mkdir()
    load_dst = _make_temp_settings_yaml(src, load_dir, load_atlas=tmp_path / 'a.osa')
    content = load_dst.read_text()
    assert f'System.LoadAtlasFromFile: "{tmp_path / "a.osa"}"' in content
    assert 'System.SaveAtlasToFile' not in content


def test_quat_to_se3_identity() -> None:
    """Identity quaternion should yield identity SE3."""
    mat = _quat_to_se3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    np.testing.assert_array_almost_equal(mat, np.eye(4))


def test_parse_and_reconcile_trajectory_aligns_and_marks_lost(tmp_path: pathlib.Path) -> None:
    """
    Trajectory entries should land in their corresponding frame slot.

    Missing frames must end up with is_lost=True and an all-NaN pose.
    """
    n = 6
    frame_ts = 1000.0 + np.arange(n, dtype=np.float64) / 60.0
    tracked = np.array([False, False, True, True, True, True])
    traj_path = tmp_path / 'traj.txt'
    _make_euroc_trajectory(traj_path, frame_ts, tracked)

    poses, is_lost = _parse_and_reconcile_trajectory(traj_path, frame_ts)

    np.testing.assert_array_equal(is_lost, ~tracked)
    # Lost rows: all NaN
    for i in range(2):
        assert np.all(np.isnan(poses[i]))
    # Tracked rows: identity rotation, translation (0.1, 0.2, 0.3)
    for i in range(2, n):
        np.testing.assert_array_almost_equal(poses[i, :3, :3], np.eye(3), decimal=5)
        np.testing.assert_array_almost_equal(poses[i, :3, 3], [0.1, 0.2, 0.3], decimal=5)


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

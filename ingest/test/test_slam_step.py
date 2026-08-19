"""Tests for the ORB-SLAM3 SLAM preprocessing step."""

from __future__ import annotations

import json
import os
import pathlib
import re
import unittest.mock as mock

import numpy as np
import pytest
import zarr
from numcodecs import Blosc
from polyumi_ingest.preproc.slam_step import (
    OrbSlam3Step,
    _SLAM_MASK_PNG,
    _downsample_settings,
    _export_telemetry_json,
    _make_temp_settings_yaml,
    _parse_trajectory_csv,
    _write_slam_results,
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


def _make_trajectory_csv(
    path: pathlib.Path,
    frame_ts: np.ndarray,
    tracked_mask: np.ndarray,
    frame_stride: int = 1,
) -> None:
    """
    Write a fake trajectory CSV as ``System::SaveTrajectoryCSV`` does.

    One row per frame the binary was *fed* — so with ``frame_stride`` > 1 the rows cover
    ``frame_ts[::frame_stride]`` and ``frame_idx`` counts fed frames, not source frames.
    Lost frames get a row too, flagged and zero-filled, which is what lets the parser index
    rows directly instead of matching timestamps.
    """
    t_ref = float(frame_ts[0])
    fed = np.arange(0, len(frame_ts), frame_stride)
    with open(path, 'w') as fh:
        fh.write('frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w\n')
        for k, i in enumerate(fed):
            t = float(frame_ts[i]) - t_ref
            if not tracked_mask[i]:
                fh.write(f'{k},{t:.6f},3,true,false,0,0,0,0,0,0,0\n')
            else:
                fh.write(f'{k},{t:.6f},2,false,false,0.1,0.2,0.3,0.0,0.0,0.0,1.0\n')


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

    def _fake_build(ep_grp, atlas_path, log_dir, scene_zarr):
        called_build.append(ep_grp.name)
        atlas_path.touch()

    def _fake_localize(ep_grp, episode_index, atlas_path, log_dir, scene_zarr):
        called_localize.append(ep_grp.name)
        n_frames = ep_grp['timestamps/gopro'].shape[0]
        poses = np.zeros((n_frames, 7), dtype=np.float64)
        poses[:, 6] = 1.0  # identity quaternion w=1
        from polyumi_ingest.preproc.slam_step import _write_slam_results

        _write_slam_results(ep_grp, poses, settings, atlas_path)

    with (
        mock.patch.object(step, '_build_map', side_effect=_fake_build),
        mock.patch.object(step, '_localize_episode', side_effect=_fake_localize),
    ):
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

    def _fake_build(ep_grp, atlas_path, log_dir, scene_zarr):
        atlas_path.touch()

    def _fake_localize(ep_grp, episode_index, atlas_path, log_dir, scene_zarr):
        traj_path = tmp_path / f'traj_{episode_index}.csv'
        frame_ts = np.asarray(ep_grp['timestamps/gopro'][:], dtype=np.float64)
        tracked = np.ones(n_frames, dtype=bool)
        tracked[:2] = False  # first two frames lost
        _make_trajectory_csv(traj_path, frame_ts, tracked)
        poses = _parse_trajectory_csv(traj_path, frame_ts)
        from polyumi_ingest.preproc.slam_step import _write_slam_results

        _write_slam_results(ep_grp, poses, settings, atlas_path)

    with (
        mock.patch.object(step, '_build_map', side_effect=_fake_build),
        mock.patch.object(step, '_localize_episode', side_effect=_fake_localize),
    ):
        step.run_step(scene_zarr)

    ep1 = zarr.open_group(str(scene_zarr / 'episode_1'), mode='r')

    # Array names and shapes
    assert 'gopro/slam_poses' in ep1
    assert 'gopro/slam_is_lost' not in ep1
    assert ep1['gopro/slam_poses'].shape == (n_frames, 7)
    assert ep1['gopro/slam_poses'].dtype == np.float64

    # Annotation attribute keys
    slam_attrs = ep1['annotations/slam'].attrs
    for key in (
        'n_frames_total',
        'n_frames_lost',
        'tracking_ratio',
        'n_relocalization_events',
        'orb_slam3_settings_path',
        'atlas_path',
    ):
        assert key in slam_attrs, f'Missing annotation key: {key}'

    assert int(slam_attrs['n_frames_total']) == n_frames
    assert int(slam_attrs['n_frames_lost']) == 2
    assert abs(float(slam_attrs['tracking_ratio']) - (4 / 6)) < 1e-5

    # Lost frames → all-NaN row in (N,7) array
    poses_arr = ep1['gopro/slam_poses'][:]
    for i in range(2):  # first 2 rows were lost
        assert np.all(np.isnan(poses_arr[i]))


def test_placeholder_detection_raises(tmp_path: pathlib.Path) -> None:
    """Settings YAML with placeholder values must raise before any subprocess is called."""
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    _make_episode(root, 'episode_0', session_type='MAPPING')

    yaml_with_placeholder = tmp_path / 'bad.yaml'
    yaml_with_placeholder.write_text('%YAML:1.0\nCamera.fx: 0.0  # CALIBRATE_ME\n')
    step = OrbSlam3Step(settings_yaml=yaml_with_placeholder)
    with pytest.raises(RuntimeError, match='CALIBRATE_ME'):
        step.run_step(scene_zarr)


def test_parse_trajectory_csv_maps_rows_through_the_stride(tmp_path: pathlib.Path) -> None:
    """
    Under decimation, CSV row k is source frame k*stride — not frame k.

    The binary only ever sees the kept frames and numbers them 0..M-1, so getting this
    wrong would place every pose on the wrong frame while still looking plausible.
    """
    n, stride = 12, 2
    frame_ts = 1000.0 + np.arange(n, dtype=np.float64) / 60.0
    traj_path = tmp_path / 'traj.csv'
    _make_trajectory_csv(traj_path, frame_ts, np.ones(n, dtype=bool), frame_stride=stride)

    poses = _parse_trajectory_csv(traj_path, frame_ts, frame_stride=stride)

    tracked = ~np.isnan(poses[:, 0])
    assert list(np.flatnonzero(tracked)) == list(range(0, n, stride))


def test_parse_trajectory_csv_rejects_a_wrong_stride(tmp_path: pathlib.Path) -> None:
    """
    A stride mismatch raises instead of silently shifting every pose.

    This is the failure the old timestamp matching could not catch: a whole-frame offset
    still landed inside the matching tolerance of the *wrong* frame.
    """
    n = 12
    frame_ts = 1000.0 + np.arange(n, dtype=np.float64) / 60.0
    traj_path = tmp_path / 'traj.csv'
    _make_trajectory_csv(traj_path, frame_ts, np.ones(n, dtype=bool), frame_stride=2)

    with pytest.raises(RuntimeError, match='did not decimate at the stride'):
        _parse_trajectory_csv(traj_path, frame_ts, frame_stride=1)


def test_parse_trajectory_csv_ignores_rows_past_the_grid(tmp_path: pathlib.Path) -> None:
    """The decoder can overrun the end of the mp4; those rows describe frames we don't have."""
    n = 6
    frame_ts = 1000.0 + np.arange(n, dtype=np.float64) / 60.0
    traj_path = tmp_path / 'traj.csv'
    _make_trajectory_csv(traj_path, frame_ts, np.ones(n, dtype=bool))
    with open(traj_path, 'a') as fh:  # two frames past the end, as an EOF overrun would write
        fh.write(f'{n},{n / 60.0:.6f},2,false,false,9,9,9,0,0,0,1\n')
        fh.write(f'{n + 1},{(n + 1) / 60.0:.6f},2,false,false,9,9,9,0,0,0,1\n')

    poses = _parse_trajectory_csv(traj_path, frame_ts)

    assert poses.shape == (n, 7)
    assert not np.isnan(poses[:, 0]).any()


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


def test_parse_trajectory_csv_aligns_and_marks_lost(tmp_path: pathlib.Path) -> None:
    """
    Trajectory rows should land in their corresponding frame slot.

    Frames the binary flagged lost must end up as all-NaN rows in the (N,7) pose array.
    """
    n = 6
    frame_ts = 1000.0 + np.arange(n, dtype=np.float64) / 60.0
    tracked = np.array([False, False, True, True, True, True])
    traj_path = tmp_path / 'traj.csv'
    _make_trajectory_csv(traj_path, frame_ts, tracked)

    poses = _parse_trajectory_csv(traj_path, frame_ts)

    assert poses.shape == (n, 7)
    # Lost rows: all NaN
    for i in range(2):
        assert np.all(np.isnan(poses[i]))
    # Tracked rows: translation (0.1, 0.2, 0.3), identity quaternion (0,0,0,1)
    for i in range(2, n):
        np.testing.assert_array_almost_equal(poses[i, :3], [0.1, 0.2, 0.3], decimal=5)
        np.testing.assert_array_almost_equal(poses[i, 3:], [0.0, 0.0, 0.0, 1.0], decimal=5)


# ---------------------------------------------------------------------------
# Smoke test (skipped unless POLYUMI_TEST_SCENE_DIR is set)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Downsampled settings + stride-aware bookkeeping (half-res / 30fps migration)
# ---------------------------------------------------------------------------

_SETTINGS_SAMPLE = """%YAML:1.0
Camera.type: "KannalaBrandt8"
Camera.fx: 421.4700653743
Camera.fy: 421.0212489922
Camera.cx: 674.6086093863
Camera.cy: 504.0829405907
Camera.k1: 0.0289317679
Camera.k4: -0.0436675477
Camera.width: 1352.0000000000
Camera.height: 1014.0000000000
Camera.fps: 60.0  # nominal
IMU.Frequency: 200
ORBextractor.nFeatures: 1500
"""


def _yaml_values(text: str) -> dict[str, float]:
    """Parse the simple `key: number` lines of a settings YAML."""
    out = {}
    for line in text.splitlines():
        m = re.match(r'^\s*([\w.]+)\s*:\s*([-\d.eE+]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def test_downsample_settings_scales_intrinsics_and_size() -> None:
    """Halving resolution halves fx/fy/cx/cy and the image dimensions."""
    v = _yaml_values(_downsample_settings(_SETTINGS_SAMPLE, res_div=2, fps_div=1))
    assert v['Camera.fx'] == pytest.approx(421.4700653743 / 2)
    assert v['Camera.fy'] == pytest.approx(421.0212489922 / 2)
    assert v['Camera.cx'] == pytest.approx(674.6086093863 / 2)
    assert v['Camera.cy'] == pytest.approx(504.0829405907 / 2)
    assert v['Camera.width'] == pytest.approx(676.0)
    assert v['Camera.height'] == pytest.approx(507.0)


def test_downsample_settings_leaves_distortion_coefficients_alone() -> None:
    """
    KannalaBrandt k1..k4 must NOT scale with resolution.

    They are coefficients of a polynomial in the incidence angle theta, which is
    dimensionless -- scaling them would silently corrupt the camera model in a way
    that still produces plausible-looking poses.
    """
    v = _yaml_values(_downsample_settings(_SETTINGS_SAMPLE, res_div=2, fps_div=1))
    assert v['Camera.k1'] == pytest.approx(0.0289317679)
    assert v['Camera.k4'] == pytest.approx(-0.0436675477)
    # ...and neither should unrelated settings.
    assert v['IMU.Frequency'] == pytest.approx(200)
    assert v['ORBextractor.nFeatures'] == pytest.approx(1500)


def test_downsample_settings_divides_fps_by_stride() -> None:
    """Camera.fps must reflect the rate actually fed; ORB-SLAM3 derives mMaxFrames from it."""
    v = _yaml_values(_downsample_settings(_SETTINGS_SAMPLE, res_div=1, fps_div=2))
    assert v['Camera.fps'] == pytest.approx(30.0)
    # resolution untouched when only the rate changes
    assert v['Camera.width'] == pytest.approx(1352.0)


def test_downsample_settings_is_identity_at_unity() -> None:
    """res_div=1, fps_div=1 returns the text unchanged -- the rollback path."""
    assert _downsample_settings(_SETTINGS_SAMPLE, 1, 1) == _SETTINGS_SAMPLE


def test_make_temp_settings_yaml_downsamples_and_still_injects_atlas(tmp_path: pathlib.Path) -> None:
    """Downsampling composes with the atlas-path injection rather than replacing it."""
    src = tmp_path / 'src.yaml'
    src.write_text(_SETTINGS_SAMPLE)
    dst = _make_temp_settings_yaml(src, tmp_path, load_atlas=tmp_path / 'a.osa', res_div=2, fps_div=2)
    text = dst.read_text()
    assert 'System.LoadAtlasFromFile' in text
    v = _yaml_values(text)
    assert v['Camera.width'] == pytest.approx(676.0)
    assert v['Camera.fps'] == pytest.approx(30.0)


def _slam_attrs_after_write(tmp_path: pathlib.Path, poses: np.ndarray, stride: int) -> dict:
    """Run _write_slam_results into a scratch store and return the annotations/slam attrs."""
    root = zarr.open_group(str(tmp_path / f'w{stride}.zarr'), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    _write_slam_results(ep, poses, tmp_path / 's.yaml', tmp_path / 'a.osa', frame_stride=stride)
    return dict(ep['annotations']['slam'].attrs)


def test_tracking_ratio_is_measured_over_fed_frames(tmp_path: pathlib.Path) -> None:
    """
    Under decimation the ratio counts only the frames SLAM was actually given.

    A perfect stride-2 run has a pose on every even frame and NaN on every odd one.
    Scoring that over all frames would read 50% and trip the 80% usability floor,
    condemning a flawless episode.
    """
    n = 100
    poses = np.full((n, 7), np.nan)
    poses[::2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # every fed frame tracked

    attrs = _slam_attrs_after_write(tmp_path, poses, stride=2)

    assert attrs['tracking_ratio'] == pytest.approx(1.0)
    assert attrs['frame_stride'] == 2
    assert attrs['n_frames_fed'] == 50
    # the all-frames counts stay honest alongside it
    assert attrs['n_frames_total'] == 100
    assert attrs['n_frames_lost'] == 50


def test_tracking_ratio_still_counts_losses_among_fed_frames(tmp_path: pathlib.Path) -> None:
    """A fed frame that genuinely lost tracking still lowers the ratio."""
    n = 100
    poses = np.full((n, 7), np.nan)
    poses[::2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    poses[::2][:10] = np.nan  # 10 of the 50 fed frames lost

    attrs = _slam_attrs_after_write(tmp_path, poses, stride=2)

    assert attrs['tracking_ratio'] == pytest.approx(0.8)


def test_relocalizations_are_counted_over_fed_frames(tmp_path: pathlib.Path) -> None:
    """
    A flawless stride-2 run has relocalized zero times, not once per tracked frame.

    Counted over the whole grid, ``is_lost`` alternates by construction under decimation, so
    every fed frame follows a 'lost' one and reads as a fresh relocalization — the attr came
    out ≈ the tracked count and carried no information.
    """
    n = 100
    poses = np.full((n, 7), np.nan)
    poses[::2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # every fed frame tracked

    attrs = _slam_attrs_after_write(tmp_path, poses, stride=2)

    assert attrs['n_relocalization_events'] == 0


def test_relocalizations_count_real_gaps_among_fed_frames(tmp_path: pathlib.Path) -> None:
    """Each run of tracked fed frames that follows a genuine gap counts once."""
    n = 100
    poses = np.full((n, 7), np.nan)
    poses[::2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    poses[10:20] = np.nan  # one gap: 5 fed frames lost, then tracking resumes

    attrs = _slam_attrs_after_write(tmp_path, poses, stride=2)

    assert attrs['n_relocalization_events'] == 1


def test_tracking_ratio_unchanged_at_stride_one(tmp_path: pathlib.Path) -> None:
    """
    At stride 1 fed == all frames, so the definition is identical to the old one.

    This is what keeps already-processed scenes and every consumer of the attr
    (notably polyumi_ingest.quality) valid without modification.
    """
    n = 50
    poses = np.full((n, 7), np.nan)
    poses[:40] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    attrs = _slam_attrs_after_write(tmp_path, poses, stride=1)

    assert attrs['tracking_ratio'] == pytest.approx(0.8)
    assert attrs['frame_stride'] == 1
    assert attrs['n_frames_fed'] == 50


def test_legacy_pass_arrays_are_cleared_on_rerun(tmp_path: pathlib.Path) -> None:
    """
    Re-processing a pzarr v3 store drops its slam_poses_{forward,reverse} arrays.

    Those were the un-merged passes of the old two-pass localizer. Nothing reads them now, and
    leaving them beside freshly written poses would have them silently describing a different
    run against a different atlas.
    """
    n = 8
    poses = np.zeros((n, 7))
    root = zarr.open_group(str(tmp_path / 'stale.zarr'), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    gopro = ep.require_group('gopro')
    gopro.create_array('slam_poses_forward', data=poses, compressor=_BLOSC)
    gopro.create_array('slam_poses_reverse', data=poses, compressor=_BLOSC)

    _write_slam_results(ep, poses, tmp_path / 's.yaml', tmp_path / 'a.osa')

    assert 'slam_poses' in ep['gopro']
    assert 'slam_poses_forward' not in ep['gopro']
    assert 'slam_poses_reverse' not in ep['gopro']


def test_legacy_reverse_attrs_are_cleared_on_rerun(tmp_path: pathlib.Path) -> None:
    """
    Re-processing a pzarr v3 store drops its reverse_* annotations along with its arrays.

    `reverse_pass: True` left behind next to freshly written forward-only poses reads as a
    claim about *those* poses, which would be false.
    """
    n = 8
    poses = np.zeros((n, 7))
    root = zarr.open_group(str(tmp_path / 'legacy_attrs.zarr'), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ep.require_group('annotations').require_group('slam').attrs.update(
        {'reverse_pass': True, 'reverse_merged': False, 'reverse_overlap_median_mm': 399.7}
    )

    _write_slam_results(ep, poses, tmp_path / 's.yaml', tmp_path / 'a.osa')

    attrs = dict(ep['annotations/slam'].attrs)
    assert not [k for k in attrs if k.startswith('reverse_')]
    assert attrs['n_frames_total'] == n


def test_post_chirp_counts_restrict_to_the_exported_window(tmp_path: pathlib.Path) -> None:
    """
    The gate's counts cover only frames at/after the chirp, where the export starts.

    Losses in the idle pre-chirp prefix are exactly the ones relocalization is still working
    through, and they never reach the dataset, so they must not count against the episode.
    """
    n = 20
    poses = np.zeros((n, 7))
    poses[:6] = np.nan  # lost through the idle prefix and one frame past it
    root = zarr.open_group(str(tmp_path / 'chirp.zarr'), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ts = 1_000.0 + np.arange(n, dtype=np.float64) / 60.0
    ep.require_group('timestamps').create_array('gopro', data=ts)
    ep.require_group('annotations').require_group('time_sync').attrs['gopro_chirp_end_s'] = float(ts[5])

    _write_slam_results(ep, poses, tmp_path / 's.yaml', tmp_path / 'a.osa')

    attrs = ep['annotations/slam'].attrs
    assert attrs['chirp_gated'] is True
    assert attrs['n_frames_fed'] == n  # stride 1: every frame fed
    assert attrs['n_frames_fed_post_chirp'] == 15  # frames 5..19
    assert attrs['n_frames_fed_lost_post_chirp'] == 1  # only frame 5 of those was lost
    assert attrs['n_frames_lost'] == 6  # whole-episode count still records all six


def test_post_chirp_counts_fall_back_without_the_marker(tmp_path: pathlib.Path) -> None:
    """No chirp annotation means the counts cover the whole episode, and say so."""
    n = 10
    poses = np.zeros((n, 7))
    poses[:3] = np.nan
    root = zarr.open_group(str(tmp_path / 'nochirp.zarr'), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ep.require_group('timestamps').create_array('gopro', data=1_000.0 + np.arange(n) / 60.0)

    _write_slam_results(ep, poses, tmp_path / 's.yaml', tmp_path / 'a.osa')

    attrs = ep['annotations/slam'].attrs
    assert attrs['chirp_gated'] is False
    assert attrs['n_frames_fed_post_chirp'] == n
    assert attrs['n_frames_fed_lost_post_chirp'] == 3


def test_slam_config_yaml_supplies_the_values(monkeypatch) -> None:
    """config/slam.yaml is the only home for these; env vars override it, nothing else does."""
    monkeypatch.delenv('POLYUMI_SLAM_RES_DIV', raising=False)
    monkeypatch.delenv('POLYUMI_SLAM_LOC_STRIDE', raising=False)
    monkeypatch.setattr(
        'polyumi_ingest.preproc.slam_step.load_slam_config',
        lambda: {'resolution_divisor': 4, 'localization_frame_stride': 3},
    )
    step = OrbSlam3Step()
    assert (step.resolution_divisor, step.localization_frame_stride) == (4, 3)

    # env wins over the file, for one-off experiments
    monkeypatch.setenv('POLYUMI_SLAM_LOC_STRIDE', '1')
    assert OrbSlam3Step().localization_frame_stride == 1


def test_missing_slam_config_key_raises(monkeypatch) -> None:
    """
    A key absent from slam.yaml is a hard error, not a silent in-code default.

    These two numbers decide what ORB-SLAM3 is fed and are enforced uniform across a DP
    export, so a fallback disagreeing with the config would split a corpus across two
    incompatible time bases instead of failing.
    """
    monkeypatch.delenv('POLYUMI_SLAM_RES_DIV', raising=False)
    monkeypatch.delenv('POLYUMI_SLAM_LOC_STRIDE', raising=False)
    monkeypatch.setattr(
        'polyumi_ingest.preproc.slam_step.load_slam_config',
        lambda: {'localization_frame_stride': 2},  # resolution_divisor missing
    )
    with pytest.raises(KeyError, match='resolution_divisor'):
        OrbSlam3Step()


def test_step_config_rollback(monkeypatch) -> None:
    """Env vars override defaults, and arguments override both."""
    monkeypatch.setenv('POLYUMI_SLAM_RES_DIV', '1')
    monkeypatch.setenv('POLYUMI_SLAM_LOC_STRIDE', '1')
    rollback = OrbSlam3Step()
    assert (rollback.resolution_divisor, rollback.localization_frame_stride) == (1, 1)

    explicit = OrbSlam3Step(resolution_divisor=3, localization_frame_stride=4)
    assert (explicit.resolution_divisor, explicit.localization_frame_stride) == (3, 4)

    with pytest.raises(ValueError):
        OrbSlam3Step(resolution_divisor=0)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get('POLYUMI_TEST_SCENE_DIR'),
    reason='Set POLYUMI_TEST_SCENE_DIR to a real scene directory to run this test',
)
def test_slam_step_smoke() -> None:
    """Full end-to-end run of OrbSlam3Step on a real scene directory."""
    scene_dir = pathlib.Path(os.environ['POLYUMI_TEST_SCENE_DIR'])
    step = OrbSlam3Step()
    step.run_step(scene_dir / 'scene.zarr')

    scene_zarr = scene_dir / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='r')
    episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
    episode_keys = [k for k in episodes if root[k].attrs.get('session_type') != 'MAPPING']
    assert episode_keys, 'No episode groups found after SLAM step'
    for ep_key in episode_keys:
        ep = root[ep_key]
        assert 'gopro/slam_poses' in ep
        assert 'annotations/slam' in ep


def test_settings_yaml_carries_the_gripper_mask(tmp_path: pathlib.Path) -> None:
    """
    The mask path must reach ORB-SLAM3 through the generated settings YAML.

    Neither PolyUMI binary has a CLI flag for it (both already overload their positional
    argv), so the YAML is the only channel. If this key stops being emitted, SLAM silently
    tracks on unmasked frames -- which does not crash, it just quietly produces a map
    nothing can relocalize against.
    """
    # Note the source is deliberately not named settings.yaml: the function writes
    # <tmp_dir>/settings.yaml, so a same-named source in the same directory is its own output.
    src = tmp_path / 'camera.yaml'
    src.write_text('%YAML:1.0\nCamera.fx: 400.0\n')
    mask = tmp_path / 'slam_mask.png'

    with_mask = tmp_path / 'with'
    without_mask = tmp_path / 'without'
    default_mask = tmp_path / 'default'
    for d in (with_mask, without_mask, default_mask):
        d.mkdir()

    assert f'Mask.Path: "{mask}"' in _make_temp_settings_yaml(src, with_mask, mask_png=mask).read_text()

    # Masked by default, so a caller that never thought about it (view_slam) still reproduces
    # what production does. Opting out has to be deliberate -- the C++ treats an absent key as
    # legal, so a forgotten mask would otherwise be a silent unmasked run.
    assert f'Mask.Path: "{_SLAM_MASK_PNG}"' in _make_temp_settings_yaml(src, default_mask).read_text()
    assert 'Mask.Path' not in _make_temp_settings_yaml(src, without_mask, mask_png=None).read_text()


def test_shipped_gripper_mask_is_binary_and_masks_the_bottom() -> None:
    """
    Guard the shipped mask's polarity, which is silently invertible.

    Non-zero means *discarded*: the C++ does setTo(0, mask), so a mask saved inverted would
    blank the scene and keep the gripper -- strictly worse than no mask, and it looks fine
    until you read a SLAM log. The gripper sits along the bottom edge of the fisheye, so a
    correctly-signed mask covers far more of the last row than the first.
    """
    import cv2

    m = cv2.imread(str(_SLAM_MASK_PNG), cv2.IMREAD_GRAYSCALE)
    assert m is not None, f'{_SLAM_MASK_PNG} is missing or unreadable'
    assert set(np.unique(m)).issubset({0, 255}), 'mask must be binary'

    masked = m > 0
    assert masked[-1].mean() > 0.9, 'bottom row should be almost entirely gripper'
    assert masked[0].mean() < 0.5, 'top row is scene, not gripper -- mask may be inverted'
    assert 0.15 < masked.mean() < 0.55, f'masked {masked.mean():.1%} of frame, implausible'

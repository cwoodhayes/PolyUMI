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
    _merge_forward_reverse,
    _parse_and_reconcile_trajectory,
    _parse_decoded_frame_count,
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
        traj_path = tmp_path / f'traj_{episode_index}.txt'
        frame_ts = np.asarray(ep_grp['timestamps/gopro'][:], dtype=np.float64)
        tracked = np.ones(n_frames, dtype=bool)
        tracked[:2] = False  # first two frames lost
        _make_euroc_trajectory(traj_path, frame_ts, tracked)
        poses = _parse_and_reconcile_trajectory(traj_path, frame_ts)
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
    yaml_with_placeholder = tmp_path / 'bad.yaml'
    yaml_with_placeholder.write_text('%YAML:1.0\nCamera.fx: 0.0  # CALIBRATE_ME\n')
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


def test_parse_and_reconcile_trajectory_aligns_and_marks_lost(tmp_path: pathlib.Path) -> None:
    """
    Trajectory entries should land in their corresponding frame slot.

    Missing frames must end up as all-NaN rows in the (N,7) pose array.
    """
    n = 6
    frame_ts = 1000.0 + np.arange(n, dtype=np.float64) / 60.0
    tracked = np.array([False, False, True, True, True, True])
    traj_path = tmp_path / 'traj.txt'
    _make_euroc_trajectory(traj_path, frame_ts, tracked)

    poses = _parse_and_reconcile_trajectory(traj_path, frame_ts)

    assert poses.shape == (n, 7)
    # Lost rows: all NaN
    for i in range(2):
        assert np.all(np.isnan(poses[i]))
    # Tracked rows: translation (0.1, 0.2, 0.3), identity quaternion (0,0,0,1)
    for i in range(2, n):
        np.testing.assert_array_almost_equal(poses[i, :3], [0.1, 0.2, 0.3], decimal=5)
        np.testing.assert_array_almost_equal(poses[i, 3:], [0.0, 0.0, 0.0, 1.0], decimal=5)


def _make_euroc_trajectory_reversed(
    path: pathlib.Path,
    frame_ts: np.ndarray,
    tracked_mask: np.ndarray,
    anchor_idx: int | None = None,
) -> None:
    """
    Write a fake EuRoC trajectory as the localizer's *reverse* sweep would.

    The reverse sweep runs frames back-to-front on a flipped video clock
    ``tframe' = t_anchor - (frame_ts - frame_ts[0])``, so timestamps still
    increase as ORB-SLAM3 requires; see
    ``mono_inertial_gopro_vi_localize.cc``'s ``RunLocalizationPass``. This
    mirrors that transform so ``reversed_clock=True`` can be tested without
    the real binary.

    ``anchor_idx`` is the index of the last frame the binary decoded (the frame
    its clock is flipped about); it defaults to the last frame of the grid.
    Pass a smaller index to simulate a decode that stopped early.
    """
    t0 = float(frame_ts[0])
    idx = len(frame_ts) - 1 if anchor_idx is None else anchor_idx
    anchor_video = float(frame_ts[idx] - t0)
    with open(path, 'w') as fh:
        for i, ts in enumerate(frame_ts):
            if not tracked_mask[i]:
                continue
            t_ns = (anchor_video - (float(ts) - t0)) * 1e9
            fh.write(f'{t_ns:.6f} 0.1 0.2 0.3 0.0 0.0 0.0 1.0\n')


def test_parse_and_reconcile_trajectory_reversed_clock_lands_on_original_frames(
    tmp_path: pathlib.Path,
) -> None:
    """A reverse-pass trajectory (flipped clock) reconciles back onto the original frame grid."""
    n = 6
    frame_ts = 1000.0 + np.arange(n, dtype=np.float64) / 60.0
    tracked = np.array([True, True, True, True, False, False])  # reverse loses the tail
    traj_path = tmp_path / 'traj_rev.txt'
    _make_euroc_trajectory_reversed(traj_path, frame_ts, tracked)

    poses = _parse_and_reconcile_trajectory(traj_path, frame_ts, reversed_clock=True)

    assert poses.shape == (n, 7)
    for i in range(4):
        np.testing.assert_array_almost_equal(poses[i, :3], [0.1, 0.2, 0.3], decimal=5)
    for i in range(4, n):
        assert np.all(np.isnan(poses[i]))


def test_parse_and_reconcile_trajectory_reversed_clock_honours_explicit_anchor(
    tmp_path: pathlib.Path,
) -> None:
    """
    A reverse pass anchored short of the grid's end still lands on the right frames.

    The binary flips its reverse clock about the last frame it *decoded*, which
    isn't always the last frame of the grid (an early decoder stop, or a
    frame-decimating run, leaves it short).  Given that anchor, every reverse
    entry must still reconcile onto its original frame index.
    """
    n = 10
    frame_ts = 1000.0 + np.arange(n, dtype=np.float64) / 60.0
    anchor_idx = 7  # decode stopped 2 frames early
    tracked = np.zeros(n, dtype=bool)
    tracked[:8] = True  # reverse tracked everything it was given
    traj_path = tmp_path / 'traj_rev.txt'
    _make_euroc_trajectory_reversed(traj_path, frame_ts, tracked, anchor_idx=anchor_idx)

    poses = _parse_and_reconcile_trajectory(
        traj_path, frame_ts, reversed_clock=True, reverse_anchor_ts=float(frame_ts[anchor_idx])
    )

    for i in range(8):
        np.testing.assert_array_almost_equal(poses[i, :3], [0.1, 0.2, 0.3], decimal=5)
    for i in range(8, n):
        assert np.all(np.isnan(poses[i]))


def test_parse_and_reconcile_trajectory_reversed_clock_wrong_anchor_shifts_silently(
    tmp_path: pathlib.Path,
) -> None:
    """
    Regression guard: assuming frame_ts[-1] as the anchor mis-maps a short decode.

    This is why ``reverse_anchor_ts`` exists.  A whole-frame offset still falls
    inside the matching tolerance of the *wrong* frame, so the old behaviour
    produced silently shifted poses rather than skipped entries — no warning to
    notice.  Pinning that here keeps the failure mode from quietly returning.
    """
    n = 10
    frame_ts = 1000.0 + np.arange(n, dtype=np.float64) / 60.0
    anchor_idx = 7
    tracked = np.zeros(n, dtype=bool)
    tracked[:8] = True
    traj_path = tmp_path / 'traj_rev.txt'
    _make_euroc_trajectory_reversed(traj_path, frame_ts, tracked, anchor_idx=anchor_idx)

    # Default anchor (frame_ts[-1]) is wrong by 2 frames for this trajectory.
    poses = _parse_and_reconcile_trajectory(traj_path, frame_ts, reversed_clock=True)

    # Every pose is shifted 2 frames later, and nothing warns about it: frames
    # 0..1 come out empty while 8..9 get poses they should never have had.
    assert np.all(np.isnan(poses[0]))
    assert np.all(np.isnan(poses[1]))
    np.testing.assert_array_almost_equal(poses[9, :3], [0.1, 0.2, 0.3], decimal=5)
    # ...whereas the explicit anchor puts them back where they belong.
    fixed = _parse_and_reconcile_trajectory(
        traj_path, frame_ts, reversed_clock=True, reverse_anchor_ts=float(frame_ts[anchor_idx])
    )
    np.testing.assert_array_almost_equal(fixed[0, :3], [0.1, 0.2, 0.3], decimal=5)
    assert np.all(np.isnan(fixed[9]))


def test_parse_decoded_frame_count_reads_the_binarys_report(tmp_path: pathlib.Path) -> None:
    """The decoded-frame count is picked out of the localizer's stdout log."""
    log_path = tmp_path / 'slam.stdout'
    log_path.write_text(
        'Loading ORB Vocabulary...\nDecoded 405 frames into memory\n[localizer:forward] tracked 100/405 frames\n'
    )
    assert _parse_decoded_frame_count(log_path) == 405


def test_parse_decoded_frame_count_returns_none_when_absent(tmp_path: pathlib.Path) -> None:
    """A log without the line (or no log at all) yields None so callers can fall back."""
    log_path = tmp_path / 'slam.stdout'
    log_path.write_text('Loading ORB Vocabulary...\nnothing useful here\n')
    assert _parse_decoded_frame_count(log_path) is None
    assert _parse_decoded_frame_count(tmp_path / 'does_not_exist.stdout') is None


# ---------------------------------------------------------------------------
# _merge_forward_reverse
# ---------------------------------------------------------------------------


def _poses_from_valid(valid: np.ndarray, xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """(N,7) pose array: valid rows get a fixed pose at ``xyz``, others NaN."""
    n = len(valid)
    poses = np.full((n, 7), np.nan, dtype=np.float64)
    poses[valid, :3] = xyz
    poses[valid, 3:] = [0.0, 0.0, 0.0, 1.0]
    return poses


def test_merge_forward_reverse_fills_forward_gaps_with_reverse() -> None:
    """Reverse should fill exactly the frames forward lost, when both passes agree."""
    # forward lost {0,1,7}; reverse lost {5,6,7} -> overlap (both tracked) = {2,3,4};
    # forward-only-lost, reverse-recoverable = {0,1}; lost by both = {7}.
    fwd_valid = np.array([False, False, True, True, True, True, True, False])
    rev_valid = np.array([True, True, True, True, True, False, False, False])
    fwd = _poses_from_valid(fwd_valid)
    rev = _poses_from_valid(rev_valid)  # same pose everywhere -> zero disagreement in overlap

    merged, stats = _merge_forward_reverse(fwd, rev)

    assert not np.any(np.isnan(merged[0]))  # forward lost, reverse filled it
    assert not np.any(np.isnan(merged[1]))
    for i in range(2, 7):
        assert not np.any(np.isnan(merged[i]))  # tracked by at least one pass
    assert np.all(np.isnan(merged[7]))  # lost by both — stays NaN
    assert stats['merged'] is True
    assert stats['n_reverse_filled'] == 2
    assert stats['n_overlap'] == 3
    assert stats['overlap_median_mm'] == pytest.approx(0.0, abs=1e-6)


def test_merge_forward_reverse_never_overwrites_a_tracked_forward_frame() -> None:
    """Even where reverse disagrees, a forward-tracked frame keeps its forward pose."""
    fwd_valid = np.array([True, True, True])
    rev_valid = np.array([True, True, True])
    fwd = _poses_from_valid(fwd_valid, xyz=(0.0, 0.0, 0.0))
    rev = _poses_from_valid(rev_valid, xyz=(1.0, 0.0, 0.0))  # 1m off — well past the guard

    merged, stats = _merge_forward_reverse(fwd, rev)

    # Guard trips (1m >> 50mm threshold) -> reverse discarded, forward unchanged.
    np.testing.assert_array_almost_equal(merged[:, :3], fwd[:, :3])
    assert stats['merged'] is False
    assert stats['overlap_median_mm'] == pytest.approx(1000.0, abs=1.0)


def test_merge_forward_reverse_no_reverse_tracking_is_a_noop() -> None:
    """If the reverse pass tracked nothing, the merge returns forward unchanged."""
    fwd_valid = np.array([True, False, True])
    fwd = _poses_from_valid(fwd_valid)
    rev = np.full((3, 7), np.nan)

    merged, stats = _merge_forward_reverse(fwd, rev)

    np.testing.assert_array_equal(merged, fwd)
    assert stats['n_reverse'] == 0
    assert stats['n_reverse_filled'] == 0


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

"""Tests for the eef-pose preprocessing step and its body-frame transform."""

import pathlib

import numpy as np
import pytest
import zarr
from scipy.spatial.transform import RigidTransform, Rotation

from polyumi_ingest.config import load_gripper_calib
from polyumi_ingest.preproc import EefPoseStep
from polyumi_ingest.preproc.eef_pose_step import _max_pose_jump_m
from polyumi_ingest.transforms import gopro_to_hand_transform, retarget_body_frame

# A body offset with both a translation and a rotation, in the same ballpark as the real
# gripper calibration (~20 cm), so the tests exercise the lever arm rather than a near-identity.
_T_X_TARGET = RigidTransform.from_components(
    translation=np.array([-0.049, 0.21, 0.0]),
    rotation=Rotation.from_quat([0.7071068, 0, 0.7071068, 0]),
)


def _pose_array(tf: RigidTransform) -> np.ndarray:
    """Flatten a (possibly stacked) RigidTransform to (N,7) [x y z qx qy qz qw]."""
    return np.concatenate([np.atleast_2d(tf.translation), np.atleast_2d(tf.rotation.as_quat())], axis=1)


def test_retarget_body_frame_round_trip() -> None:
    """Re-targeting a sensor trajectory onto a body frame, then back, must be exact."""
    rng = np.random.default_rng(0)
    n = 16
    T_w_x = RigidTransform.from_components(translation=rng.normal(size=(n, 3)), rotation=Rotation.random(n, rng=rng))
    sensor = _pose_array(T_w_x)

    target = retarget_body_frame(sensor, _T_X_TARGET)
    back = retarget_body_frame(target, _T_X_TARGET.inv())

    np.testing.assert_allclose(back[:, :3], T_w_x.translation, atol=1e-9)
    rot_err = (Rotation.from_quat(back[:, 3:]).inv() * T_w_x.rotation).magnitude()
    assert rot_err.max() < 1e-9


def test_retarget_body_frame_matches_composition() -> None:
    """The output is exactly T_w_x · T_x_target, not merely self-consistent."""
    rng = np.random.default_rng(1)
    n = 8
    T_w_x = RigidTransform.from_components(translation=rng.normal(size=(n, 3)), rotation=Rotation.random(n, rng=rng))
    expected = T_w_x * _T_X_TARGET

    out = retarget_body_frame(_pose_array(T_w_x), _T_X_TARGET)

    np.testing.assert_allclose(out[:, :3], expected.translation, atol=1e-9)
    rot_err = (Rotation.from_quat(out[:, 3:]).inv() * expected.rotation).magnitude()
    assert rot_err.max() < 1e-9


def test_retarget_body_frame_preserves_nan() -> None:
    """Rows the pose source could not solve (SLAM tracking loss) stay NaN instead of raising."""
    sensor = _pose_array(RigidTransform.from_components(translation=np.zeros((4, 3)), rotation=Rotation.identity(4)))
    sensor[2] = np.nan

    out = retarget_body_frame(sensor, _T_X_TARGET)

    assert np.isnan(out[2]).all()
    assert not np.isnan(out[[0, 1, 3]]).any()


def test_retarget_body_frame_all_nan() -> None:
    """An episode with no solved poses returns all-NaN rather than raising."""
    out = retarget_body_frame(np.full((3, 7), np.nan), _T_X_TARGET)
    assert out.shape == (3, 7)
    assert np.isnan(out).all()


def test_gopro_to_hand_transform_requires_calibration() -> None:
    """A calibration missing T_gopro_to_fingertip raises rather than silently defaulting."""
    with pytest.raises(KeyError, match='T_gopro_to_fingertip'):
        gopro_to_hand_transform({'some_other_key': {}})


def test_gopro_to_hand_transform_reads_calibration() -> None:
    """T_gopro_to_fingertip is parsed as a pose of the hand expressed in the GoPro frame."""
    tf = gopro_to_hand_transform(
        {'T_gopro_to_fingertip': {'translation': [0.0, 0.0, 0.072], 'rotation': [0.0, 0.0, 0.0, 1.0]}}
    )
    np.testing.assert_allclose(tf.translation, [0.0, 0.0, 0.072], atol=1e-12)


def test_body_frame_offset_does_not_cancel_under_relative_pose() -> None:
    """
    The reason this step exists: a body-frame offset survives the relative representation.

    The policy trains on inv(T_0)·T_k. A shared *world* frame cancels there, but a *body*
    offset X conjugates it — inv(T_0·X)·(T_k·X) = inv(X)·(inv(T_0)·T_k)·X — leaking a
    (R - I)·x translation error. This test pins that the error is large enough to matter,
    so nobody "simplifies" the step away.
    """
    # A pure 30-degree wrist rotation, no translation of the hand frame itself.
    T_w_hand_0 = RigidTransform.from_components(translation=np.zeros(3), rotation=Rotation.identity())
    T_w_hand_k = RigidTransform.from_components(
        translation=np.zeros(3), rotation=Rotation.from_rotvec(np.radians(30) * np.array([0, 0, 1.0]))
    )

    # Correct: relative motion of the hand frame is a pure rotation.
    rel_correct = T_w_hand_0.inv() * T_w_hand_k
    np.testing.assert_allclose(rel_correct.translation, np.zeros(3), atol=1e-12)

    # Wrong: relative motion measured at an offset body frame picks up the lever arm.
    rel_wrong = (T_w_hand_0 * _T_X_TARGET).inv() * (T_w_hand_k * _T_X_TARGET)
    assert np.linalg.norm(rel_wrong.translation) > 0.10  # >10 cm of phantom translation


def _build_scene(tmp_path: pathlib.Path, *, with_optitrack: bool, with_slam: bool) -> pathlib.Path:
    """Build a minimal one-episode scene.zarr with the requested pose sources."""
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    root.attrs['n_episodes'] = 1
    ep = root.create_group('episode_0')

    n = 10
    gopro_ts = np.arange(n, dtype=np.float64) / 60.0
    ep.create_group('timestamps').create_array('gopro', data=gopro_ts)

    if with_slam:
        # SLAM reports the GoPro frame directly: a 0→1 m slide along x, no rotation.
        T_s_gp = RigidTransform.from_components(
            translation=np.linspace(0, 1, n)[:, None] * np.array([1.0, 0, 0]),
            rotation=Rotation.identity(n),
        )
        ep.create_group('gopro').create_array('slam_poses', data=_pose_array(T_s_gp))
        # Step 2 always writes this alongside the poses, and step 5 adds max_pose_jump_m to it.
        ep.require_group('annotations').create_group('slam').attrs['frame_stride'] = 1

    if with_optitrack:
        opti_ts = np.arange(2 * n, dtype=np.float64) / 120.0  # a faster, offset grid
        opti = root.create_group('optitrack')
        opti.create_array('timestamps', data=opti_ts)
        opti.create_array(
            'pose',
            data=_pose_array(
                RigidTransform.from_components(translation=np.zeros((2 * n, 3)), rotation=Rotation.identity(2 * n))
            ),
        )
    return scene_zarr


def test_eef_pose_step_writes_both_alternates_when_available(tmp_path: pathlib.Path) -> None:
    """With both sources present, both eef/pose_<source> arrays are written on the gopro grid."""
    scene_zarr = _build_scene(tmp_path, with_optitrack=True, with_slam=True)

    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    assert ep['eef/pose_optitrack'].shape == (10, 7)  # resampled onto the 10-frame gopro grid
    assert ep['eef/pose_slam'].shape == (10, 7)
    assert ep['eef/pose_optitrack'].attrs['world_frame'] == 'optitrack'
    assert ep['eef/pose_slam'].attrs['world_frame'] == 'slam'
    assert ep['eef/pose_optitrack'].attrs['body_frame'] == 'hand'
    assert ep['eef'].attrs['available_sources'] == ['optitrack', 'slam']
    assert ep['eef'].attrs['default_source'] == 'optitrack'  # _SOURCE_PREFERENCE order
    assert ep['eef'].attrs['body_frame'] == 'hand'


def test_eef_pose_step_writes_only_slam_when_optitrack_absent(tmp_path: pathlib.Path) -> None:
    """Without optitrack, only eef/pose_slam is written; the GoPro→hand hop is applied to it."""
    scene_zarr = _build_scene(tmp_path, with_optitrack=False, with_slam=True)

    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    assert 'eef/pose_optitrack' not in ep
    assert ep['eef'].attrs['available_sources'] == ['slam']
    assert ep['eef'].attrs['default_source'] == 'slam'
    assert ep['eef/pose_slam'].attrs['body_frame'] == 'hand'
    assert ep['eef/pose_slam'].attrs['n_nan'] == 0

    # SLAM poses here have identity rotation, so the hand offset applies unrotated and the
    # expected trajectory is the slide plus T_gp_hand's translation. Derived from the live
    # calibration rather than hardcoded, so recalibrating doesn't spuriously fail this.
    T_gp_hand = gopro_to_hand_transform(load_gripper_calib())
    expected = np.linspace(0, 1, 10)[:, None] * np.array([1.0, 0, 0]) + T_gp_hand.translation
    np.testing.assert_allclose(ep['eef/pose_slam'][:][:, :3], expected, atol=1e-9)


def test_eef_pose_step_writes_only_optitrack_when_slam_absent(tmp_path: pathlib.Path) -> None:
    """Without slam, only eef/pose_optitrack is written."""
    scene_zarr = _build_scene(tmp_path, with_optitrack=True, with_slam=False)

    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    assert 'eef/pose_slam' not in ep
    assert ep['eef'].attrs['available_sources'] == ['optitrack']
    assert ep['eef'].attrs['default_source'] == 'optitrack'


def test_eef_pose_step_skips_episode_without_source(tmp_path: pathlib.Path) -> None:
    """An episode with neither source is skipped rather than failing the whole scene."""
    scene_zarr = _build_scene(tmp_path, with_optitrack=False, with_slam=False)

    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    assert 'eef' not in ep


def test_eef_pose_step_is_idempotent_without_force(tmp_path: pathlib.Path) -> None:
    """Re-running leaves existing eef/pose_<source> alone unless force is set."""
    scene_zarr = _build_scene(tmp_path, with_optitrack=False, with_slam=True)
    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='a')
    ep['eef/pose_slam'][0, 0] = 42.0  # sentinel

    EefPoseStep().run_step(scene_zarr)
    assert zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')['eef/pose_slam'][0, 0] == 42.0

    EefPoseStep().run_step(scene_zarr, force=True)
    assert zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')['eef/pose_slam'][0, 0] != 42.0


def test_eef_pose_step_force_adds_newly_available_source(tmp_path: pathlib.Path) -> None:
    """
    --force recomputes even when only a new source became available since the last run.

    Without force, a scene that gained an optitrack recording after an earlier slam-only
    eef-pose run should still need --force to pick it up (available_sources doesn't
    superset the newly-available set until forced).
    """
    scene_zarr = _build_scene(tmp_path, with_optitrack=False, with_slam=True)
    EefPoseStep().run_step(scene_zarr)
    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    assert ep['eef'].attrs['available_sources'] == ['slam']

    # optitrack becomes available (e.g. re-synced into the scene)
    opti_ts = np.arange(20, dtype=np.float64) / 120.0
    opti = zarr.open_group(str(scene_zarr), mode='a').require_group('optitrack')
    opti.create_array('timestamps', data=opti_ts)
    opti.create_array(
        'pose',
        data=_pose_array(RigidTransform.from_components(translation=np.zeros((20, 3)), rotation=Rotation.identity(20))),
    )

    EefPoseStep().run_step(scene_zarr, force=True)
    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    assert set(ep['eef'].attrs['available_sources']) == {'optitrack', 'slam'}
    assert 'eef/pose_optitrack' in ep


# ---------------------------------------------------------------------------
# max_pose_jump_m — the measurement behind quality.py's pose-jump check
# ---------------------------------------------------------------------------


def _jump_episode(
    tmp_path: pathlib.Path,
    positions: np.ndarray,
    *,
    stride: int = 1,
    chirp_end_s: float | None = None,
) -> zarr.Group:
    """Build an episode group carrying ``positions`` as a hand-frame trajectory at 60 Hz."""
    n = len(positions)
    ep = zarr.open_group(str(tmp_path / 'ep.zarr'), mode='w', zarr_format=2)
    ep.create_group('timestamps').create_array('gopro', data=np.arange(n, dtype=np.float64) / 60.0)
    ep.require_group('annotations').require_group('slam').attrs['frame_stride'] = stride
    if chirp_end_s is not None:
        ep['annotations'].require_group('time_sync').attrs['gopro_chirp_end_s'] = chirp_end_s
    return ep


def _poses(positions: np.ndarray) -> np.ndarray:
    """(N,7) poses at ``positions`` with identity rotation; NaN rows stay NaN."""
    out = np.full((len(positions), 7), np.nan)
    valid = ~np.isnan(positions).any(axis=1)
    out[valid, :3] = positions[valid]
    out[valid, 3:] = [0.0, 0.0, 0.0, 1.0]
    return out


def test_max_pose_jump_measures_the_largest_adjacent_step(tmp_path: pathlib.Path) -> None:
    """Smooth motion reports its own step size; one teleport reports the teleport."""
    pos = np.zeros((10, 3))
    pos[:, 0] = np.arange(10) * 0.01  # a steady 1 cm per frame
    ep = _jump_episode(tmp_path, pos)
    assert _max_pose_jump_m(ep, _poses(pos)) == pytest.approx(0.01)

    pos[5:, 0] += 1.14  # the relocalization teleport
    assert _max_pose_jump_m(ep, _poses(pos)) == pytest.approx(1.15, abs=1e-6)


def test_max_pose_jump_ignores_pairs_spanning_a_gap(tmp_path: pathlib.Path) -> None:
    """
    A pair straddling lost frames covers more ground legitimately and must not count.

    How far the hand moved while tracking was down says nothing about a bad pose; the
    lost-frame threshold is what judges gaps.
    """
    pos = np.zeros((10, 3))
    pos[:, 0] = np.arange(10) * 0.01
    pos[4:7] = np.nan  # tracking lost, then resumes 3 cm further along
    ep = _jump_episode(tmp_path, pos)

    assert _max_pose_jump_m(ep, _poses(pos)) == pytest.approx(0.01)


def test_max_pose_jump_walks_the_fed_grid_under_decimation(tmp_path: pathlib.Path) -> None:
    """
    At stride 2 the odd frames have no pose, so consecutive *fed* frames are 2 apart.

    Treating every row as adjacent would compare a real pose against a NaN one and find
    nothing at all to measure.
    """
    pos = np.full((10, 3), np.nan)
    pos[::2] = 0.0
    pos[::2, 0] = np.arange(5) * 0.02  # 2 cm between fed frames
    ep = _jump_episode(tmp_path, pos, stride=2)

    assert _max_pose_jump_m(ep, _poses(pos)) == pytest.approx(0.02)


def test_max_pose_jump_skips_the_pre_chirp_prefix(tmp_path: pathlib.Path) -> None:
    """
    The idle prefix is where the localizer is still settling and never reaches the dataset.

    Judged over the whole recording a jump there would condemn an episode whose exported
    span is clean — the same reasoning that puts the frame counts on the post-chirp window.
    """
    pos = np.zeros((10, 3))
    pos[:, 0] = np.arange(10) * 0.01
    pos[:3, 0] += 2.0  # a wild prefix, settled by the time the chirp ends
    ep = _jump_episode(tmp_path, pos, chirp_end_s=4.0 / 60.0)

    assert _max_pose_jump_m(ep, _poses(pos)) == pytest.approx(0.01)


def test_max_pose_jump_is_none_without_two_adjacent_tracked_frames(tmp_path: pathlib.Path) -> None:
    """Nothing measurable is None, not zero — zero would read as a perfectly smooth episode."""
    pos = np.full((10, 3), np.nan)
    pos[0] = [0.0, 0.0, 0.0]
    ep = _jump_episode(tmp_path, pos)

    assert _max_pose_jump_m(ep, _poses(pos)) is None


def test_eef_pose_step_records_the_jump_beside_the_slam_metrics(tmp_path: pathlib.Path) -> None:
    """
    The step stores the measurement; quality.py turns it into a verdict at read time.

    Into ``annotations/slam``, not onto the pose array: that is the mapping
    ``auto_unusable_reasons`` is handed, so no consumer has to merge two sources.
    """
    scene_zarr = _build_scene(tmp_path, with_optitrack=False, with_slam=True)

    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    # The fixture slides 0->1 m over 10 frames, and a pure translation carries the hand
    # frame with it unchanged, so each step is 1/9 m.
    assert ep['annotations/slam'].attrs['max_pose_jump_m'] == pytest.approx(1 / 9)
    assert 'max_pose_jump_m' not in ep['eef/pose_slam'].attrs

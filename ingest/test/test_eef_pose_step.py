"""Tests for the eef-pose preprocessing step and its body-frame transform."""

import pathlib

import numpy as np
import pytest
import zarr
from scipy.spatial.transform import RigidTransform, Rotation

from polyumi_ingest.config import load_gripper_calib
from polyumi_ingest.preproc import EefPoseStep
from polyumi_ingest.transforms import gopro_to_hand_transform, retarget_body_frame

# A body offset with both a translation and a rotation, in the same ballpark as the real
# gripper calibration (~20 cm), so the tests exercise the lever arm rather than a near-identity.
_T_X_TARGET = RigidTransform.from_components(
    translation=np.array([-0.049, 0.21, 0.0]),
    rotation=Rotation.from_quat([0.7071068, 0, 0.7071068, 0]),
)


def _pose_array(tf: RigidTransform) -> np.ndarray:
    """Flatten a (possibly stacked) RigidTransform to (N,7) [x y z qx qy qz qw]."""
    return np.concatenate(
        [np.atleast_2d(tf.translation), np.atleast_2d(tf.rotation.as_quat())], axis=1
    )


def test_retarget_body_frame_round_trip() -> None:
    """Re-targeting a sensor trajectory onto a body frame, then back, must be exact."""
    rng = np.random.default_rng(0)
    n = 16
    T_w_x = RigidTransform.from_components(
        translation=rng.normal(size=(n, 3)), rotation=Rotation.random(n, rng=rng)
    )
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
    T_w_x = RigidTransform.from_components(
        translation=rng.normal(size=(n, 3)), rotation=Rotation.random(n, rng=rng)
    )
    expected = T_w_x * _T_X_TARGET

    out = retarget_body_frame(_pose_array(T_w_x), _T_X_TARGET)

    np.testing.assert_allclose(out[:, :3], expected.translation, atol=1e-9)
    rot_err = (Rotation.from_quat(out[:, 3:]).inv() * expected.rotation).magnitude()
    assert rot_err.max() < 1e-9


def test_retarget_body_frame_preserves_nan() -> None:
    """Rows the pose source could not solve (SLAM tracking loss) stay NaN instead of raising."""
    sensor = _pose_array(
        RigidTransform.from_components(
            translation=np.zeros((4, 3)), rotation=Rotation.identity(4)
        )
    )
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
    """A calibration missing T_gopro_to_hand raises rather than silently defaulting."""
    with pytest.raises(KeyError, match='T_gopro_to_hand'):
        gopro_to_hand_transform({'some_other_key': {}})


def test_gopro_to_hand_transform_reads_calibration() -> None:
    """T_gopro_to_hand is parsed as a pose of the hand expressed in the GoPro frame."""
    tf = gopro_to_hand_transform(
        {'T_gopro_to_hand': {'translation': [0.0, 0.0, 0.072], 'rotation': [0.0, 0.0, 0.0, 1.0]}}
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
    T_w_hand_0 = RigidTransform.from_components(
        translation=np.zeros(3), rotation=Rotation.identity()
    )
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

    if with_optitrack:
        opti_ts = np.arange(2 * n, dtype=np.float64) / 120.0  # a faster, offset grid
        opti = root.create_group('optitrack')
        opti.create_array('timestamps', data=opti_ts)
        opti.create_array(
            'pose',
            data=_pose_array(
                RigidTransform.from_components(
                    translation=np.zeros((2 * n, 3)), rotation=Rotation.identity(2 * n)
                )
            ),
        )
    return scene_zarr


def test_eef_pose_step_prefers_optitrack_and_resamples(tmp_path: pathlib.Path) -> None:
    """With both sources present, optitrack wins and lands on the gopro grid."""
    scene_zarr = _build_scene(tmp_path, with_optitrack=True, with_slam=True)

    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    assert ep['eef/pose'].shape == (10, 7)  # resampled onto the 10-frame gopro grid
    assert ep['eef'].attrs['source'] == 'optitrack'
    assert ep['eef'].attrs['world_frame'] == 'optitrack'
    assert ep['eef'].attrs['body_frame'] == 'hand'


def test_eef_pose_step_falls_back_to_slam(tmp_path: pathlib.Path) -> None:
    """Without optitrack, slam is used and the GoPro→hand hop is applied to it."""
    scene_zarr = _build_scene(tmp_path, with_optitrack=False, with_slam=True)

    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    assert ep['eef'].attrs['source'] == 'slam'
    assert ep['eef'].attrs['body_frame'] == 'hand'
    assert ep['eef'].attrs['n_nan'] == 0

    # SLAM poses here have identity rotation, so the hand offset applies unrotated and the
    # expected trajectory is the slide plus T_gp_hand's translation. Derived from the live
    # calibration rather than hardcoded, so recalibrating doesn't spuriously fail this.
    T_gp_hand = gopro_to_hand_transform(load_gripper_calib())
    expected = np.linspace(0, 1, 10)[:, None] * np.array([1.0, 0, 0]) + T_gp_hand.translation
    np.testing.assert_allclose(ep['eef/pose'][:][:, :3], expected, atol=1e-9)


def test_eef_pose_step_skips_episode_without_source(tmp_path: pathlib.Path) -> None:
    """An episode with neither source is skipped rather than failing the whole scene."""
    scene_zarr = _build_scene(tmp_path, with_optitrack=False, with_slam=False)

    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')
    assert 'eef/pose' not in ep


def test_eef_pose_step_is_idempotent_without_force(tmp_path: pathlib.Path) -> None:
    """Re-running leaves existing eef/pose alone unless force is set."""
    scene_zarr = _build_scene(tmp_path, with_optitrack=False, with_slam=True)
    EefPoseStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr / 'episode_0'), mode='a')
    ep['eef/pose'][0, 0] = 42.0  # sentinel

    EefPoseStep().run_step(scene_zarr)
    assert zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')['eef/pose'][0, 0] == 42.0

    EefPoseStep().run_step(scene_zarr, force=True)
    assert zarr.open_group(str(scene_zarr / 'episode_0'), mode='r')['eef/pose'][0, 0] != 42.0

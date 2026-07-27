"""Tests for the world/optitrack/slam static-transform tree exported to /tf_static."""

import json
import pathlib

import numpy as np
import zarr
from mcap.reader import make_reader

from polyumi_ingest.export.mcap import export_episode_to_mcap


def _build_episode(tmp_path: pathlib.Path, *, with_optitrack: bool, with_slam: bool) -> tuple[zarr.Group, zarr.Group]:
    """Build a minimal 2-frame episode, optionally with SLAM poses and/or scene-level optitrack data."""
    n = 2
    H, W = 32, 48
    fps = 60.0

    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')

    finger_frames = np.zeros((n, H, W, 3), dtype=np.uint8)
    finger_audio = np.zeros(64, dtype=np.float32)
    finger_ts = np.arange(n, dtype=np.float64) / 10.0
    audio_ts = np.arange(64, dtype=np.float64) / 16_000.0
    ep.create_group('finger').create_array('frames', data=finger_frames)
    ep['finger'].create_array('finger_piezo', data=finger_audio)  # type: ignore[union-attr]
    ep['finger'].create_array('finger_air', data=finger_audio)  # type: ignore[union-attr]
    ts_grp = ep.create_group('timestamps')
    ts_grp.create_array('finger', data=finger_ts)
    ts_grp.create_array('finger_piezo', data=audio_ts)
    ts_grp.create_array('finger_air', data=audio_ts)

    gopro_frames = np.zeros((n, H, W, 3), dtype=np.uint8)
    gopro_ts = np.arange(n, dtype=np.float64) / fps
    gopro_grp = ep.create_group('gopro')
    gopro_grp.create_array('frames', data=gopro_frames)
    ts_grp.create_array('gopro', data=gopro_ts)

    if with_slam:
        slam_poses = np.zeros((n, 7), dtype=np.float64)
        slam_poses[:, 6] = 1.0  # identity quaternion (qw=1)
        gopro_grp.create_array('slam_poses', data=slam_poses)

    if with_optitrack:
        _identity = {'translation': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0, 1.0]}
        root.attrs['gripper_calib'] = {
            'T_gripper_base_to_optitrack_rigid_body': _identity,
            'T_gripper_base_to_gopro': _identity,
            'T_optitrack_to_world': _identity,
        }
        ot_ts = np.arange(n, dtype=np.float64) / 10.0
        ot_poses = np.zeros((n, 7), dtype=np.float64)
        ot_poses[:, 6] = 1.0
        opt_grp = root.create_group('optitrack')
        opt_grp.create_array('pose', data=ot_poses)
        opt_grp.create_array('timestamps', data=ot_ts)

    return root, ep


def _tf_static_messages(mcap_path: pathlib.Path) -> list[dict]:
    with mcap_path.open('rb') as f:
        reader = make_reader(f)
        return [json.loads(msg.data) for _, _, msg in reader.iter_messages(topics=['/tf_static'])]


def test_identity_world_to_slam_transform_when_no_optitrack(tmp_path: pathlib.Path) -> None:
    """Without optitrack, an identity world -> slam transform is published directly."""
    root, ep = _build_episode(tmp_path, with_optitrack=False, with_slam=True)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    tfs = _tf_static_messages(mcap_path)
    assert len(tfs) == 1
    tf = tfs[0]
    assert tf['parent_frame_id'] == 'world'
    assert tf['child_frame_id'] == 'slam'
    assert tf['translation'] == {'x': 0.0, 'y': 0.0, 'z': 0.0}
    assert tf['rotation'] == {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}


def test_no_extra_world_to_slam_transform_when_optitrack_present(tmp_path: pathlib.Path) -> None:
    """With optitrack, the chain is world -> optitrack -> slam; no direct world -> slam shortcut."""
    root, ep = _build_episode(tmp_path, with_optitrack=True, with_slam=True)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    tfs = _tf_static_messages(mcap_path)
    pairs = {(tf['parent_frame_id'], tf['child_frame_id']) for tf in tfs}
    assert pairs == {('world', 'optitrack'), ('optitrack', 'slam')}


def test_no_tf_static_messages_without_optitrack_or_slam(tmp_path: pathlib.Path) -> None:
    """With neither optitrack nor SLAM, no static transforms are published at all."""
    root, ep = _build_episode(tmp_path, with_optitrack=False, with_slam=False)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    assert _tf_static_messages(mcap_path) == []

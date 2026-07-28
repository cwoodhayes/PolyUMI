"""Tests for the /eef/pose_optitrack and /eef/pose_slam Foxglove channels."""

import json
import pathlib

import numpy as np
import zarr
from mcap.reader import make_reader

from polyumi_ingest.export.mcap import export_episode_to_mcap


def _build_episode(tmp_path: pathlib.Path, *, with_eef_optitrack: bool, with_eef_slam: bool) -> zarr.Group:
    """Build a minimal 2-frame episode, optionally with eef/pose_<source> hand-frame arrays."""
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

    if with_eef_optitrack or with_eef_slam:
        eef_grp = ep.create_group('eef')
        if with_eef_optitrack:
            pose = np.zeros((n, 7), dtype=np.float64)
            pose[:, 6] = 1.0  # identity quaternion (qw=1)
            eef_grp.create_array('pose_optitrack', data=pose)
        if with_eef_slam:
            pose = np.zeros((n, 7), dtype=np.float64)
            pose[:, 6] = 1.0
            pose[0, :3] = np.nan  # one lost frame, to confirm it's skipped like /slam/pose
            eef_grp.create_array('pose_slam', data=pose)

    return root, ep  # type: ignore[return-value]


def _messages(mcap_path: pathlib.Path, topic: str) -> list[dict]:
    with mcap_path.open('rb') as f:
        reader = make_reader(f)
        return [json.loads(msg.data) for _, _, msg in reader.iter_messages(topics=[topic])]


def test_eef_pose_channels_absent_when_step_5_has_not_run(tmp_path: pathlib.Path) -> None:
    """No eef group at all — the new channels simply aren't registered, non-fatal."""
    root, ep = _build_episode(tmp_path, with_eef_optitrack=False, with_eef_slam=False)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    assert _messages(mcap_path, '/eef/pose_optitrack') == []
    assert _messages(mcap_path, '/eef/pose_slam') == []


def test_eef_pose_optitrack_channel_uses_optitrack_frame_id(tmp_path: pathlib.Path) -> None:
    """/eef/pose_optitrack is followable and frame_id'd against the optitrack world frame."""
    root, ep = _build_episode(tmp_path, with_eef_optitrack=True, with_eef_slam=False)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    msgs = _messages(mcap_path, '/eef/pose_optitrack')
    assert len(msgs) == 2
    assert all(m['frame_id'] == 'optitrack' for m in msgs)
    assert _messages(mcap_path, '/eef/pose_slam') == []


def test_eef_pose_slam_channel_skips_nan_and_uses_slam_frame_id(tmp_path: pathlib.Path) -> None:
    """/eef/pose_slam is frame_id'd against slam and skips the NaN (lost) frame."""
    root, ep = _build_episode(tmp_path, with_eef_optitrack=False, with_eef_slam=True)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    msgs = _messages(mcap_path, '/eef/pose_slam')
    assert len(msgs) == 1  # one of the two frames is NaN and skipped
    assert msgs[0]['frame_id'] == 'slam'


def test_both_eef_pose_channels_present_when_both_sources_available(tmp_path: pathlib.Path) -> None:
    """With both eef/pose_<source> arrays, both channels are followable side by side."""
    root, ep = _build_episode(tmp_path, with_eef_optitrack=True, with_eef_slam=True)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    assert len(_messages(mcap_path, '/eef/pose_optitrack')) == 2
    assert len(_messages(mcap_path, '/eef/pose_slam')) == 1

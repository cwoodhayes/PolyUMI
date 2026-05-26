"""Smoke test for ArUco ImageAnnotations export in mcap.py."""

import json
import pathlib

import numpy as np
import zarr
from mcap.reader import make_reader

from polyumi_ingest.export.mcap import export_episode_to_mcap


def _build_minimal_episode(tmp_path: pathlib.Path) -> tuple[zarr.Group, zarr.Group]:
    """Synthesize a 2-frame scene with finger streams + gopro frames + finger_corners."""
    n_gopro = 2
    n_finger = 2
    H, W = 32, 48
    fps = 60.0

    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')

    # Finger streams (required by mcap exporter).
    finger_frames = np.zeros((n_finger, H, W, 3), dtype=np.uint8)
    finger_audio = np.zeros(64, dtype=np.float32)
    finger_ts = np.arange(n_finger, dtype=np.float64) / 10.0
    audio_ts = np.arange(64, dtype=np.float64) / 16_000.0
    ep.create_group('finger').create_array('frames', data=finger_frames)
    ep['finger'].create_array('finger_piezo', data=finger_audio)  # type: ignore[union-attr]
    ep['finger'].create_array('finger_air', data=finger_audio)  # type: ignore[union-attr]
    ts_grp = ep.create_group('timestamps')
    ts_grp.create_array('finger', data=finger_ts)
    ts_grp.create_array('finger_piezo', data=audio_ts)
    ts_grp.create_array('finger_air', data=audio_ts)

    # GoPro frames + matching timestamps.
    gopro_frames = np.zeros((n_gopro, H, W, 3), dtype=np.uint8)
    gopro_ts = np.arange(n_gopro, dtype=np.float64) / fps
    ep.create_group('gopro').create_array('frames', data=gopro_frames)
    ts_grp.create_array('gopro', data=gopro_ts)

    # finger_corners: frame 0 has the left marker, frame 1 is empty.
    finger_corners = np.full((n_gopro, 2, 4, 2), np.nan, dtype=np.float32)
    finger_corners[0, 0] = np.array(
        [
            [10.0, 10.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0],
        ],
        dtype=np.float32,
    )
    # Dense per-frame width series: frame 0 has a value, frame 1 is NaN.
    width_m = np.array([0.042, np.nan], dtype=np.float32)

    gw_grp = ep.create_group('annotations').create_group('gripper_width')
    gw_grp.create_array('finger_corners', data=finger_corners)
    gw_grp.create_array('width_m', data=width_m)
    gw_grp.attrs['left_id'] = 0
    gw_grp.attrs['right_id'] = 1

    return root, ep


def test_aruco_annotations_channel_written(tmp_path: pathlib.Path) -> None:
    """Exporter publishes /gopro/aruco_annotations with one populated and one empty message."""
    root, ep = _build_minimal_episode(tmp_path)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    with mcap_path.open('rb') as f:
        reader = make_reader(f)
        topics = {ch.topic for ch in reader.get_summary().channels.values()}  # type: ignore[union-attr]
        assert '/gopro/aruco_annotations' in topics

        messages = [
            (schema, channel, msg) for schema, channel, msg in reader.iter_messages(topics=['/gopro/aruco_annotations'])
        ]
    assert len(messages) == 2

    payload0 = json.loads(messages[0][2].data)
    payload1 = json.loads(messages[1][2].data)

    assert len(payload0['points']) == 1
    assert len(payload0['texts']) == 1
    assert payload0['points'][0]['type'] == 2  # LINE_LOOP
    assert len(payload0['points'][0]['points']) == 4
    assert payload0['texts'][0]['text'] == '0'

    assert payload1['points'] == []
    assert payload1['texts'] == []


def test_gripper_width_channel_written(tmp_path: pathlib.Path) -> None:
    """Exporter publishes /gripper/width with one message per non-NaN width sample."""
    root, ep = _build_minimal_episode(tmp_path)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    with mcap_path.open('rb') as f:
        reader = make_reader(f)
        topics = {ch.topic for ch in reader.get_summary().channels.values()}  # type: ignore[union-attr]
        assert '/gripper/width' in topics

        messages = [
            (schema, channel, msg) for schema, channel, msg in reader.iter_messages(topics=['/gripper/width'])
        ]
    # Frame 0 has a width, frame 1 is NaN and gets skipped.
    assert len(messages) == 1
    payload = json.loads(messages[0][2].data)
    assert abs(payload['width_m'] - 0.042) < 1e-6

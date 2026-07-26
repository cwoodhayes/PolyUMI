"""Tests for the ArUco gripper-width preprocessing step."""

import pathlib

import cv2
import numpy as np
import pytest
import zarr
from polyumi_ingest.preproc import ArucoGripperWidthStep


def test_aruco_step_no_markers(tmp_path: pathlib.Path) -> None:
    """With blank frames the step runs cleanly and reports zero detections."""
    n_frames = 5
    H, W = 240, 320
    timestamps = np.arange(n_frames, dtype=np.float64) / 60.0

    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ep.attrs['session_dir'] = 'session_0'
    ep.create_group('timestamps').create_array('gopro', data=timestamps)

    # v3: GoPro frames are decoded from the gopro.mp4 sidecar. Blank frames → no markers.
    mp4 = tmp_path / 'session_0' / 'gopro.mp4'
    mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
    if not writer.isOpened():
        pytest.skip('cv2.VideoWriter (mp4v) unavailable in this environment')
    for _ in range(n_frames):
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    writer.release()

    ArucoGripperWidthStep().run(scene_zarr)

    root = zarr.open_group(str(scene_zarr), mode='r')
    out_grp = root['episode_0/annotations/gripper_width']
    width_m = np.asarray(out_grp['width_m'][:])  # type: ignore[index]

    assert width_m.shape == (n_frames,)
    assert np.isnan(width_m).all()
    assert float(out_grp.attrs['detection_rate']) == 0.0  # type: ignore[arg-type]
    assert int(out_grp.attrs['n_detected']) == 0  # type: ignore[arg-type]
    assert int(out_grp.attrs['n_frames']) == n_frames  # type: ignore[arg-type]
    assert root.attrs['preprocessing_steps'] == [4]

    finger_corners = np.asarray(out_grp['finger_corners'][:])  # type: ignore[index]
    assert finger_corners.shape == (n_frames, 2, 4, 2)
    assert np.isnan(finger_corners).all()

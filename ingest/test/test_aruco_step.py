"""Tests for the ArUco gripper-width preprocessing step."""

import pathlib

import numpy as np
import zarr
from polyumi_ingest.preproc import ArucoGripperWidthStep


def test_aruco_step_no_markers(tmp_path: pathlib.Path) -> None:
    """With blank frames the step runs cleanly and reports zero detections."""
    n_frames = 5
    H, W = 240, 320
    frames = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
    timestamps = np.arange(n_frames, dtype=np.float64) / 60.0

    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ep.create_group('gopro').create_array('frames', data=frames)
    ep.create_group('timestamps').create_array('gopro', data=timestamps)

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

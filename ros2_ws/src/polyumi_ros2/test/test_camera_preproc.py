"""
The inference camera0_rgb transform must match the ingest exporter's contract.

Pure unit test — no rclpy — so it runs under plain pytest and colcon alike.
"""

import cv2
import numpy as np

from polyumi_ros2.camera_preproc import (
    CAMERA0_RGB_INTERPOLATION,
    CAMERA0_RGB_RESOLUTION,
    resize_camera0_rgb,
)


def test_contract_matches_exporter() -> None:
    """224x224 + INTER_AREA — identical to polyumi_ingest.camera_preproc."""
    assert CAMERA0_RGB_RESOLUTION == 224
    assert CAMERA0_RGB_INTERPOLATION == cv2.INTER_AREA


def test_resize_shape_and_interpolation() -> None:
    """resize_camera0_rgb returns (224,224,3) uint8 via INTER_AREA, byte-for-byte."""
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)

    out = resize_camera0_rgb(frame)

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    expected = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    assert np.array_equal(out, expected)

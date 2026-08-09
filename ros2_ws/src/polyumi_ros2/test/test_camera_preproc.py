"""
The inference camera0_rgb transform must match the ingest exporter's contract.

Pure unit test — no rclpy — so it runs under plain pytest and colcon alike.
"""

import cv2
import numpy as np

from polyumi_ros2.camera_preproc import (
    CAMERA0_RGB_INTERPOLATION,
    CAMERA0_RGB_RESOLUTION,
    SOURCE_ASPECT,
    crop_to_source_aspect,
    resize_camera0_rgb,
)


def test_contract_matches_exporter() -> None:
    """224x224 + INTER_AREA + the 4:3 crop — identical to polyumi_ingest.camera_preproc."""
    assert CAMERA0_RGB_RESOLUTION == 224
    assert CAMERA0_RGB_INTERPOLATION == cv2.INTER_AREA
    assert SOURCE_ASPECT == 4 / 3


def test_resize_shape_and_interpolation() -> None:
    """resize_camera0_rgb returns (224,224,3) uint8 via INTER_AREA, byte-for-byte."""
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(1080, 1440, 3), dtype=np.uint8)  # already 4:3, so no crop

    out = resize_camera0_rgb(frame)

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    expected = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    assert np.array_equal(out, expected)


def test_hdmi_pillarbox_is_cropped_to_the_recorded_field_of_view() -> None:
    """
    The live Elgato frame is the GoPro's 4:3 image pillarboxed into 1080p — bars must not survive.

    Measured on hardware: content occupies columns 240..1679 exactly. Without this the policy
    sees a frame a quarter of which is black, and the real content squeezed into 3/4 the width —
    which the training frames never were. It still runs; it just runs on the wrong pixels.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, 240:1680] = 200

    cropped = crop_to_source_aspect(frame)

    assert cropped.shape == (1080, 1440, 3)
    assert cropped.min() == 200, 'a black bar survived the crop'


def test_gopro_recording_frames_are_not_cropped() -> None:
    """2704x2028 is already 4:3 — the exporter side must stay a no-op."""
    frame = np.zeros((2028, 2704, 3), dtype=np.uint8)
    assert crop_to_source_aspect(frame).shape == frame.shape

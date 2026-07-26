"""The camera0_rgb preprocessing contract must match the inference node's transform."""

import cv2
import numpy as np

from polyumi_ingest.camera_preproc import (
    CAMERA0_RGB_INTERPOLATION,
    CAMERA0_RGB_RESOLUTION,
    resize_camera0_rgb,
)


def test_contract_constants() -> None:
    """The contract pins 224x224 and INTER_AREA (anti-aliased downscale)."""
    assert CAMERA0_RGB_RESOLUTION == 224
    assert CAMERA0_RGB_INTERPOLATION == cv2.INTER_AREA


def test_resize_shape_dtype_and_interpolation() -> None:
    """resize_camera0_rgb returns (224,224,3) uint8 via INTER_AREA, byte-for-byte."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(540, 960, 3), dtype=np.uint8)  # 1080p-ish RGB

    out = resize_camera0_rgb(frame)

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    expected = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    assert np.array_equal(out, expected)


def test_resize_preserves_channel_order() -> None:
    """A pure-red RGB frame stays red (channel order untouched by the resize)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[..., 0] = 255  # R

    out = resize_camera0_rgb(frame)

    assert out[..., 0].min() == 255
    assert out[..., 1].max() == 0
    assert out[..., 2].max() == 0

"""The camera0_rgb preprocessing contract must match the inference node's transform."""

import hashlib

import cv2
import numpy as np
import pytest

# Shared verbatim with the ROS-side suite, which loads this same file by path — see its docstring.
from camera0_rgb_golden import GOLDEN_VECTORS, golden_frame

from polyumi_ingest.camera_preproc import (
    CAMERA0_RGB_INTERPOLATION,
    CAMERA0_RGB_RESOLUTION,
    MAX_BAR_INTENSITY,
    SOURCE_ASPECT,
    crop_to_source_aspect,
    discarded_bar_intensity,
    resize_camera0_rgb,
)


def test_contract_constants() -> None:
    """The contract pins 224x224, INTER_AREA (anti-aliased downscale), and the 4:3 crop."""
    assert CAMERA0_RGB_RESOLUTION == 224
    assert CAMERA0_RGB_INTERPOLATION == cv2.INTER_AREA
    assert SOURCE_ASPECT == 4 / 3


def test_resize_shape_dtype_and_interpolation() -> None:
    """resize_camera0_rgb returns (224,224,3) uint8 via INTER_AREA, byte-for-byte."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(600, 800, 3), dtype=np.uint8)  # already 4:3, so no crop

    out = resize_camera0_rgb(frame)

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    expected = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    assert np.array_equal(out, expected)


def test_gopro_recording_frames_are_not_cropped() -> None:
    """2704x2028 is already 4:3 — cropping it would silently change every exported dataset."""
    frame = np.zeros((2028, 2704, 3), dtype=np.uint8)
    assert crop_to_source_aspect(frame).shape == frame.shape


def test_hdmi_pillarbox_is_cropped_to_the_recorded_field_of_view() -> None:
    """
    The GoPro pillarboxes its 4:3 image into the 1080p HDMI frame; the bars must not reach 224².

    Measured on hardware: content occupies columns 240..1679 exactly. Getting this wrong is the
    quiet kind of failure — the policy still runs, on frames a quarter of which are black.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, 240:1680] = 200  # the real image, bars either side

    cropped = crop_to_source_aspect(frame)

    assert cropped.shape == (1080, 1440, 3)
    assert cropped.min() == 200, 'a black bar survived the crop'


def test_letterboxed_input_is_cropped_the_other_way() -> None:
    """A source taller than 4:3 gets its top/bottom bars dropped, not squashed in."""
    frame = np.zeros((1000, 800, 3), dtype=np.uint8)
    frame[200:800] = 200  # 800x600 of content, 4:3

    cropped = crop_to_source_aspect(frame)

    assert cropped.shape == (600, 800, 3)
    assert cropped.min() == 200


def test_black_pillarbox_bars_read_as_bars() -> None:
    """The expected case: the discarded columns are the HDMI black the crop assumes they are."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, 240:1680] = 200

    assert discarded_bar_intensity(frame) <= MAX_BAR_INTENSITY


def test_a_real_16_9_frame_is_flagged_rather_than_silently_narrowed() -> None:
    """
    The failure the check exists for: a genuine 16:9 source loses a quarter of its FOV, quietly.

    On this side that would be a non-4:3 gopro.mp4, and the narrowing is baked into the exported
    dataset permanently rather than merely being served to a running policy.
    """
    frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)  # content edge to edge, no bars

    assert discarded_bar_intensity(frame) == pytest.approx(200.0)
    assert discarded_bar_intensity(frame) > MAX_BAR_INTENSITY


def test_no_crop_means_nothing_discarded() -> None:
    """A 4:3 frame discards nothing, so the check must not invent an intensity for it."""
    assert discarded_bar_intensity(np.full((2028, 2704, 3), 200, dtype=np.uint8)) == 0.0


def test_resize_preserves_channel_order() -> None:
    """A pure-red RGB frame stays red (channel order untouched by the resize)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[..., 0] = 255  # R

    out = resize_camera0_rgb(frame)

    assert out[..., 0].min() == 255
    assert out[..., 1].max() == 0
    assert out[..., 2].max() == 0


@pytest.mark.parametrize(('h', 'w', 'expected'), GOLDEN_VECTORS)
def test_golden_vector_is_stable_across_environments(h: int, w: int, expected: str) -> None:
    """
    The two real source resolutions must produce these exact bytes, in either Python environment.

    The expectations live in camera0_rgb_golden.py, which the ROS suite reads by path, so this
    assertion and its inference-side twin cannot drift apart.
    """
    out = resize_camera0_rgb(golden_frame(h, w))

    assert hashlib.sha256(out.tobytes()).hexdigest() == expected, (
        'the camera0_rgb transform changed. Unless that was deliberate, training and inference no '
        'longer agree on pixels — see docs/data-format.md ("camera0_rgb preprocessing contract").'
    )

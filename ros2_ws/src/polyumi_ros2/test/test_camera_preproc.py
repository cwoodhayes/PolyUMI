"""
The inference camera transforms must match the ingest exporter's contracts.

Pure unit test — no rclpy — so it runs under plain pytest and colcon alike.
"""

import hashlib
import importlib.util
import pathlib

import cv2
import numpy as np
import pytest

from polyumi_ros2.camera_preproc import (
    CAMERA0_RGB_INTERPOLATION,
    CAMERA0_RGB_RESOLUTION,
    MAX_BAR_INTENSITY,
    SOURCE_ASPECT,
    crop_finger_rgb,
    crop_to_source_aspect,
    discarded_bar_intensity,
    resize_camera0_rgb,
)

# The golden vectors are shared with the ingest suite, and there is exactly one copy of them —
# two copies of a digest could drift apart silently and would then be asserting nothing. Loaded
# by path rather than imported because this package runs in the ROS venv (Python 3.12) and cannot
# import anything from polyumi_ingest (Python >= 3.13); the file itself needs only numpy. See its
# docstring, and docs/data-format.md ("Camera preprocessing contracts").
# Resolved from __file__, not cwd, so `colcon test` and a bare `pytest test/` both find it.
_GOLDEN = pathlib.Path(__file__).resolve().parents[4] / 'ingest' / 'test' / 'camera_preproc_golden.py'
if not _GOLDEN.is_file():
    raise FileNotFoundError(
        f'camera preprocessing golden vectors not found at {_GOLDEN}. They are shared with the ingest test '
        'suite; if that file moved, update this path rather than forking a second copy.'
    )
_spec = importlib.util.spec_from_file_location('camera_preproc_golden', _GOLDEN)
_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_golden)
CAMERA0_GOLDEN_VECTORS = _golden.CAMERA0_GOLDEN_VECTORS
FINGER_GOLDEN_VECTORS = _golden.FINGER_GOLDEN_VECTORS
FINGER_GOLDEN_CROP = _golden.FINGER_GOLDEN_CROP
golden_frame = _golden.golden_frame


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


def test_black_pillarbox_bars_read_as_bars() -> None:
    """The expected case: the discarded columns are the HDMI black the crop assumes they are."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, 240:1680] = 200

    assert discarded_bar_intensity(frame) <= MAX_BAR_INTENSITY


def test_a_real_16_9_frame_is_flagged_rather_than_silently_narrowed() -> None:
    """
    The failure the check exists for: a genuine 16:9 source loses a quarter of its FOV, quietly.

    Same invisible train/inference skew the crop fixes, pointing the other way — the policy still
    runs, on a narrower view than it trained on.
    """
    frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)  # content edge to edge, no bars

    assert discarded_bar_intensity(frame) == pytest.approx(200.0)
    assert discarded_bar_intensity(frame) > MAX_BAR_INTENSITY


def test_no_crop_means_nothing_discarded() -> None:
    """A 4:3 frame discards nothing, so the check must not invent an intensity for it."""
    assert discarded_bar_intensity(np.full((1080, 1440, 3), 200, dtype=np.uint8)) == 0.0


@pytest.mark.parametrize(('h', 'w', 'expected'), CAMERA0_GOLDEN_VECTORS)
def test_golden_vector_is_stable_across_environments(h: int, w: int, expected: str) -> None:
    """
    The two real source resolutions must produce these exact bytes, in either Python environment.

    Same expectations the ingest suite asserts, read from the same file — so this passing while
    that one fails (or vice versa) means the two environments genuinely disagree on pixels, not
    that someone updated one copy of a constant.
    """
    out = resize_camera0_rgb(golden_frame(h, w))

    assert hashlib.sha256(out.tobytes()).hexdigest() == expected, (
        'the camera0_rgb transform changed. Unless that was deliberate, training and inference no '
        'longer agree on pixels — see docs/data-format.md ("Camera preprocessing contracts").'
    )


def test_finger_crop_drops_the_occluded_strip() -> None:
    """The shipped crop takes the real 1152x648 view down to 982x648, keeping the right-hand columns."""
    frame = golden_frame(648, 1152)

    out = crop_finger_rgb(frame, **FINGER_GOLDEN_CROP)

    assert out.shape == (648, 982, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out, frame[:, 170:])


def test_finger_crop_none_bounds_mean_the_frame_edge() -> None:
    """
    ``None`` resolves against the frame, not against a hardcoded resolution.

    That is what lets the same config survive a change of finger camera: only the occluded strip
    is a fixed number of pixels, and the far edges are wherever the sensor ends.
    """
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    assert crop_finger_rgb(frame, x_min=10).shape == (100, 190, 3)
    assert crop_finger_rgb(frame, y_min=5).shape == (95, 200, 3)
    assert crop_finger_rgb(frame).shape == frame.shape


@pytest.mark.parametrize(
    'bounds',
    [
        {'x_min': -1},
        {'x_min': 200},
        {'x_min': 50, 'x_max': 50},
        {'x_min': 60, 'x_max': 50},
        {'x_max': 201},
        {'y_min': 100},
        {'y_max': 101},
    ],
)
def test_finger_crop_rejects_bounds_outside_the_frame(bounds: dict) -> None:
    """
    A crop configured for a different camera resolution must fail loudly, not export a sliver.

    Silently clipping would produce a plausible-looking image that is not the one the contract
    names — the same invisible skew the golden vectors exist to prevent.
    """
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match='finger crop'):
        crop_finger_rgb(frame, **bounds)


def test_finger_crop_returns_a_copy_not_a_view() -> None:
    """The inference side reads frames out of buffers that get recycled underneath it."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    out = crop_finger_rgb(frame, x_min=10)
    out[:] = 255

    assert frame.max() == 0


def test_finger_crop_resizes_after_cropping() -> None:
    """
    ``output_size`` is (width, height) and is applied to the *cropped* frame, via INTER_AREA.

    Deliberately non-square, so a width/height transposition cannot pass.
    """
    frame = golden_frame(648, 1152)

    out = crop_finger_rgb(frame, output_size=(200, 100), **FINGER_GOLDEN_CROP)

    assert out.shape == (100, 200, 3)
    expected = cv2.resize(frame[:, 170:], (200, 100), interpolation=cv2.INTER_AREA)
    assert np.array_equal(out, expected)


def test_finger_crop_preserves_channel_order() -> None:
    """A pure-red RGB frame stays red — a BGR slip here would swap it in every exported frame."""
    frame = np.zeros((648, 1152, 3), dtype=np.uint8)
    frame[..., 0] = 255  # R

    out = crop_finger_rgb(frame, output_size=(224, 224), **FINGER_GOLDEN_CROP)

    assert out[..., 0].min() == 255
    assert out[..., 1].max() == 0
    assert out[..., 2].max() == 0


@pytest.mark.parametrize(('h', 'w', 'output_size', 'expected'), FINGER_GOLDEN_VECTORS)
def test_finger_golden_vector_is_stable_across_environments(
    h: int, w: int, output_size: tuple[int, int] | None, expected: str
) -> None:
    """
    The finger crop must produce these exact bytes, in either Python environment.

    The crop-only entry is a pure array slice, so its digest is really a check that the two
    *implementations* still agree; the resized entry additionally pins ``cv2.INTER_AREA``, which
    is a different C++ implementation on each side. Regenerate only when a changed transform is
    the intent — an existing checkpoint is tied to the crop it trained under.
    """
    out = crop_finger_rgb(golden_frame(h, w), output_size=output_size, **FINGER_GOLDEN_CROP)

    assert hashlib.sha256(out.tobytes()).hexdigest() == expected, (
        'the finger_rgb transform changed. Unless that was deliberate, training and inference no '
        'longer agree on pixels — see docs/data-format.md ("Camera preprocessing contracts").'
    )

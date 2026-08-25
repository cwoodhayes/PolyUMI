"""
The camera preprocessing contracts, shared by export and inference.

A policy compares like with like or not at all: the frames the DP exporter bakes into the
dataset and the frames the inference node feeds the policy must go through the *same* pixel
transforms, or we introduce train/inference skew. This module is the single source of truth for
them on the ingest side; the ROS inference node (``ros2_ws/.../camera_preproc.py``)
reimplements the identical contracts because the two packages cannot share a Python import (uv
workspace vs. ROS venv). **Keep the whole file in lock-step with that copy** — see
``docs/data-format.md`` ("Camera preprocessing contracts").

Two contracts live here, both taking an **RGB** ``(H,W,3)`` uint8 frame and returning uint8.
Any ``float32/255`` normalization is applied downstream (the training loader / inference node),
not here — the exported store stays uint8 per the UMI convention.

* ``camera0_rgb`` (:func:`resize_camera0_rgb`) — the GoPro. Centre-cropped to the 4:3 recording
  aspect (a no-op on a frame already at that aspect), then squashed to ``(224,224,3)`` with
  ``cv2.INTER_AREA``, the correct anti-aliased choice for downscaling.
* ``finger_rgb`` (:func:`crop_finger_rgb`) — the finger camera. Cropped to given bounds, since
  the mount occludes a fixed strip of its view, with an optional resize. Its geometry comes from
  ``ingest/config/finger_camera.yaml``; the ROS side has no consumer for it yet (the node grows
  a finger-camera subscription only once the clock-domain issue in
  ``ros2_ws/.../config/inference.yaml`` is resolved), and it is mirrored now so that wiring is a
  subscription rather than a second derivation of the transform.
"""

import cv2
import numpy as np

#: Output side length of the policy's ``camera0_rgb`` observation (shape ``[3, 224, 224]``).
CAMERA0_RGB_RESOLUTION = 224
#: Interpolation for the resize — INTER_AREA anti-aliases when downscaling.
CAMERA0_RGB_INTERPOLATION = cv2.INTER_AREA
#: Aspect ratio the GoPro records at (2704x2028), and the aspect every frame is cropped to
#: before the squash. See :func:`crop_to_source_aspect`.
SOURCE_ASPECT = 4 / 3
#: Above this mean intensity the pixels the crop discards are not a black bar, so the crop is
#: eating real image. Loose: HDMI black sits a little above 0 and the capture card adds noise,
#: while the failure being caught is gross (a quarter of a real scene, not a dim bar).
MAX_BAR_INTENSITY = 16.0


def crop_to_source_aspect(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Centre-crop a frame to the GoPro's 4:3 recording aspect.

    Training frames come from ``gopro.mp4`` at 2704x2028, which is already 4:3 — this is a no-op
    on them. Inference frames come off the Elgato at 1920x1080, and the GoPro's clean-HDMI output
    **pillarboxes** that same 4:3 image into 16:9: measured on hardware, the content occupies
    columns 240..1679 exactly, with pure black bars either side. So the field of view is
    identical; without this crop the inference frame would carry 480 columns of black the policy
    never saw in training, and squeeze the real content into 3/4 of the width.

    Cropping rather than letterbox-padding is what keeps the two identical: it recovers precisely
    the 1440x1080 the camera framed, so both paths squash the same field of view.
    """
    h, w = frame_rgb.shape[:2]
    crop_w = round(h * SOURCE_ASPECT)
    if crop_w < w:  # pillarboxed (or otherwise too wide) — drop the side bars
        x0 = (w - crop_w) // 2
        return frame_rgb[:, x0 : x0 + crop_w]
    crop_h = round(w / SOURCE_ASPECT)
    if crop_h < h:  # letterboxed — drop the top/bottom bars
        y0 = (h - crop_h) // 2
        return frame_rgb[y0 : y0 + crop_h]
    return frame_rgb


def discarded_bar_intensity(frame_rgb: np.ndarray) -> float:
    """
    Mean channel intensity (0-255) of the pixels :func:`crop_to_source_aspect` would throw away.

    The crop assumes anything wider than 4:3 is a pillarbox, i.e. that the columns it drops are
    black bars. If that assumption is ever wrong — a GoPro configured to a genuine 16:9 mode, a
    capture card doing its own scaling, a ``gopro.mp4`` recorded at some other aspect — the crop
    silently removes a quarter of the real field of view instead. That is the same invisible
    train/inference skew the crop exists to fix, just pointing the other way: the policy keeps
    running, on pixels nobody chose.

    So: sample this once and compare against :data:`MAX_BAR_INTENSITY`. Returns 0.0 when the crop
    is a no-op, since nothing is discarded.
    """
    kept = crop_to_source_aspect(frame_rgb)
    if kept.shape == frame_rgb.shape:
        return 0.0
    # Derived by subtraction rather than by re-deriving the crop geometry, so this cannot drift
    # out of step with the function it is checking.
    n_discarded = (frame_rgb.shape[0] * frame_rgb.shape[1] - kept.shape[0] * kept.shape[1]) * frame_rgb.shape[2]
    total = frame_rgb.sum(dtype=np.float64) - kept.sum(dtype=np.float64)
    return float(total / n_discarded)


def resize_camera0_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    """Crop to 4:3, then resize onto the camera0_rgb grid (224x224, INTER_AREA)."""
    return cv2.resize(
        crop_to_source_aspect(frame_rgb),
        (CAMERA0_RGB_RESOLUTION, CAMERA0_RGB_RESOLUTION),
        interpolation=CAMERA0_RGB_INTERPOLATION,
    )


#: Interpolation for the optional finger-camera resize — INTER_AREA, as ``camera0_rgb`` uses.
FINGER_RGB_INTERPOLATION = cv2.INTER_AREA


def crop_finger_rgb(
    frame_rgb: np.ndarray,
    *,
    x_min: int = 0,
    x_max: int | None = None,
    y_min: int = 0,
    y_max: int | None = None,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Crop a finger-camera frame to its useful region, optionally resizing the result.

    Unlike :func:`crop_to_source_aspect`, whose geometry is derived from the frame's aspect, this
    crop is *given* — the gripper mount occludes a fixed strip of the finger camera's view, and
    which strip is a property of the hardware, not of the image. The bounds are half-open
    ``[min, max)``; ``None`` means the frame's own edge, so the geometry survives a change of
    camera resolution rather than silently landing somewhere else.

    ``output_size`` is ``(width, height)``, or ``None`` to keep the crop at its native size. The
    exporter's default is ``None``: the crop is then an exact array slice, identical in any numpy,
    and the choice of encoder input size stays with whoever builds the policy.

    Returns a contiguous copy, never a view onto ``frame_rgb`` — the inference side reads frames
    out of buffers that get recycled underneath it.

    Raises:
        ValueError: if a bound is negative, inverted, or past the edge of the frame. A crop
            configured for a different camera resolution would otherwise export a sliver of the
            intended region, which looks like a plausible image and is not one.

    """
    h, w = frame_rgb.shape[:2]
    x1 = w if x_max is None else int(x_max)
    y1 = h if y_max is None else int(y_max)
    x0, y0 = int(x_min), int(y_min)
    for axis, lo, hi, limit in (('x', x0, x1, w), ('y', y0, y1, h)):
        if not 0 <= lo < hi <= limit:
            raise ValueError(
                f'finger crop {axis}=[{lo}, {hi}) is not a non-empty range within [0, {limit}) '
                f'for a {w}x{h} frame — check the crop bounds in config/finger_camera.yaml '
                f'against the camera resolution that recorded it.'
            )
    cropped = np.ascontiguousarray(frame_rgb[y0:y1, x0:x1])
    if output_size is None:
        return cropped
    return cv2.resize(
        cropped,
        (int(output_size[0]), int(output_size[1])),
        interpolation=FINGER_RGB_INTERPOLATION,
    )

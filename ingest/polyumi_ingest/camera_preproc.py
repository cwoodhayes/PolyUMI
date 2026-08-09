"""
The camera0_rgb preprocessing contract, shared by export and inference.

A policy compares like with like or not at all: the frame the DP exporter bakes into
``camera0_rgb`` at training time and the frame the inference node feeds the policy must go
through the *same* pixel transform, or we introduce train/inference skew. This module is the
single source of truth for that transform on the ingest side; the ROS inference node
(``ros2_ws/.../policy_client_node.py``) reimplements the identical contract because the two
packages cannot share a Python import (uv workspace vs. ROS venv). Keep them in lock-step —
see ``docs/data-format.md`` ("camera0_rgb preprocessing contract").

Contract: input is an **RGB** ``(H,W,3)`` uint8 frame; it is centre-cropped to the GoPro's 4:3
recording aspect (a no-op on a frame already at that aspect) and then squashed to
``(224,224,3)`` uint8 with ``cv2.INTER_AREA`` (the correct anti-aliased choice for
downscaling). Any ``float32/255`` normalization is applied downstream (the training loader /
inference node), not here — the exported store stays uint8 per the UMI convention.
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

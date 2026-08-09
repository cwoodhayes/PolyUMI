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


def resize_camera0_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    """Crop to 4:3, then resize onto the camera0_rgb grid (224x224, INTER_AREA)."""
    return cv2.resize(
        crop_to_source_aspect(frame_rgb),
        (CAMERA0_RGB_RESOLUTION, CAMERA0_RGB_RESOLUTION),
        interpolation=CAMERA0_RGB_INTERPOLATION,
    )

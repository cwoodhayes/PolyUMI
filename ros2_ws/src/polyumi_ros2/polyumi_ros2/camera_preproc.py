"""
The camera0_rgb preprocessing contract, inference side.

This MUST stay byte-identical to the ingest exporter's
``polyumi_ingest.camera_preproc.resize_camera0_rgb``: the policy only compares like with
like, so the frame the DP exporter bakes into training and the frame this node feeds the
policy have to go through the same pixel transform. The two packages can't share a Python
import (ROS venv vs. uv workspace), so the contract is duplicated here on purpose — keep
them in lock-step. See ``docs/data-format.md`` ("camera0_rgb preprocessing contract").

Contract: RGB ``(H,W,3)`` uint8 in → centre-cropped to the GoPro's 4:3 recording aspect (a
no-op if already 4:3) → ``(224,224,3)`` uint8 out, squashed with ``cv2.INTER_AREA``
(anti-aliased downscale). The ``float32/255`` normalization is applied by the node after this,
not here.
"""

import cv2
import numpy as np

CAMERA0_RGB_RESOLUTION = 224
CAMERA0_RGB_INTERPOLATION = cv2.INTER_AREA
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

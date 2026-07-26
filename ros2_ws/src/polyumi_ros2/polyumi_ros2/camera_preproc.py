"""
The camera0_rgb preprocessing contract, inference side.

This MUST stay byte-identical to the ingest exporter's
``polyumi_ingest.camera_preproc.resize_camera0_rgb``: the policy only compares like with
like, so the frame the DP exporter bakes into training and the frame this node feeds the
policy have to go through the same pixel transform. The two packages can't share a Python
import (ROS venv vs. uv workspace), so the contract is duplicated here on purpose — keep
them in lock-step. See ``docs/data-format.md`` ("camera0_rgb preprocessing contract").

Contract: RGB ``(H,W,3)`` uint8 in → ``(224,224,3)`` uint8 out, resized with
``cv2.INTER_AREA`` (anti-aliased downscale), squashed with no crop. The ``float32/255``
normalization is applied by the node after this, not here.
"""

import cv2
import numpy as np

CAMERA0_RGB_RESOLUTION = 224
CAMERA0_RGB_INTERPOLATION = cv2.INTER_AREA


def resize_camera0_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    """Resize an RGB frame onto the camera0_rgb grid (224x224, INTER_AREA, no crop)."""
    return cv2.resize(
        frame_rgb,
        (CAMERA0_RGB_RESOLUTION, CAMERA0_RGB_RESOLUTION),
        interpolation=CAMERA0_RGB_INTERPOLATION,
    )

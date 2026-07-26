"""
The camera0_rgb preprocessing contract, shared by export and inference.

A policy compares like with like or not at all: the frame the DP exporter bakes into
``camera0_rgb`` at training time and the frame the inference node feeds the policy must go
through the *same* pixel transform, or we introduce train/inference skew. This module is the
single source of truth for that transform on the ingest side; the ROS inference node
(``ros2_ws/.../policy_client_node.py``) reimplements the identical contract because the two
packages cannot share a Python import (uv workspace vs. ROS venv). Keep them in lock-step —
see ``docs/data-format.md`` ("camera0_rgb preprocessing contract").

Contract: input is an **RGB** ``(H,W,3)`` uint8 frame; output is ``(224,224,3)`` uint8, resized
with ``cv2.INTER_AREA`` (the correct anti-aliased choice for downscaling), squashed to the
target with **no** crop. Any ``float32/255`` normalization is applied downstream (the training
loader / inference node), not here — the exported store stays uint8 per the UMI convention.
"""

import cv2
import numpy as np

#: Output side length of the policy's ``camera0_rgb`` observation (shape ``[3, 224, 224]``).
CAMERA0_RGB_RESOLUTION = 224
#: Interpolation for the resize — INTER_AREA anti-aliases when downscaling.
CAMERA0_RGB_INTERPOLATION = cv2.INTER_AREA


def resize_camera0_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    """Resize an RGB frame onto the camera0_rgb grid (224x224, INTER_AREA, no crop)."""
    return cv2.resize(
        frame_rgb,
        (CAMERA0_RGB_RESOLUTION, CAMERA0_RGB_RESOLUTION),
        interpolation=CAMERA0_RGB_INTERPOLATION,
    )

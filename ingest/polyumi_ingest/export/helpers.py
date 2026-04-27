"""Shared helpers for pzarr export functions."""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


def jpegxl_to_jpeg(frame: np.ndarray, quality: int) -> bytes:
    """Re-encode a (H, W, 3) uint8 RGB frame from pzarr JpegXL storage to JPEG bytes."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return buf.tobytes()


def encode_frames_to_jpeg(frames: np.ndarray, quality: int) -> list[bytes]:
    """
    Parallel JPEG re-encoding for a batch of RGB frames decoded from pzarr JpegXL storage.

    frames: (N, H, W, 3) uint8 RGB.
    Returns N JPEG byte strings in frame order.
    """
    with ThreadPoolExecutor() as pool:
        return list(pool.map(lambda f: jpegxl_to_jpeg(f, quality), frames))

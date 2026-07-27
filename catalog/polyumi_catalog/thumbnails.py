"""
On-demand GoPro frame thumbnails for the Episodes list + session detail pane (Phase 4).

Decodes a single representative frame directly from a session's ``gopro.mp4``
sidecar via OpenCV — independent of pzarr, so a thumbnail is available for any
synced session regardless of preprocessing stage. Each call opens its own
``cv2.VideoCapture`` and releases it before returning, so concurrent requests
(FastAPI runs sync routes in a threadpool) are safe.
"""

from __future__ import annotations

import pathlib

import cv2

# ~1s into a typical capture: cheap to seek to (decoding a handful of frames from the
# preceding keyframe, rather than the whole video), while skipping any startup black frame.
_THUMBNAIL_FRAME_INDEX = 30
_THUMBNAIL_WIDTH = 320


def session_thumbnail_jpeg(session_dir: pathlib.Path) -> bytes | None:
    """
    Return a JPEG-encoded representative frame from ``session_dir/gopro.mp4``, or ``None``.

    ``None`` covers "no gopro.mp4", "file won't open", and "no frame could be decoded"
    alike — callers should treat all of these as "no thumbnail available" rather than
    distinguishing the cause.
    """
    mp4_path = session_dir / 'gopro.mp4'
    if not mp4_path.is_file():
        return None

    cap = cv2.VideoCapture(str(mp4_path))
    try:
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = min(_THUMBNAIL_FRAME_INDEX, max(frame_count - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, bgr = cap.read()
        if not ok:
            # POS_FRAMES seeking can misbehave near a short/odd file; fall back to frame 0.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = cap.read()
        if not ok:
            return None
    finally:
        cap.release()

    h, w = bgr.shape[:2]
    if w > _THUMBNAIL_WIDTH:
        scale = _THUMBNAIL_WIDTH / w
        bgr = cv2.resize(bgr, (_THUMBNAIL_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return buf.tobytes()

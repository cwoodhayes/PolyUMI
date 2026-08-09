"""
Golden vectors for the camera0_rgb preprocessing contract — the single copy, read by both sides.

The contract itself is implemented twice, once per Python environment
(``polyumi_ingest.camera_preproc`` and ``polyumi_ros2.camera_preproc``), because the two cannot
share an import: ``polyumi_ingest`` requires Python >= 3.13 while the ROS node runs Ubuntu 24.04's
``/usr/bin/python3`` (3.12), and it would drag ``polyumi_pi``/zarr/scipy into the inference
process. See ``docs/data-format.md`` ("camera0_rgb preprocessing contract").

Identical *source* is not the guarantee that matters, though — identical *output* is. The two
environments run different library majors (measured 2026-08-09: ROS ``cv2 4.6.0`` /
``numpy 1.26.4``, the uv workspace ``cv2 4.13.0`` / ``numpy 2.4.3``), so ``cv2.resize`` is a
different C++ implementation on each side regardless of how the Python is shared. Pinning the
bytes is the only check that covers that.

So the expectations live here, once, and both test suites read *this file*: the ingest suite
imports it (pytest puts ``ingest/test`` on ``sys.path``), and
``ros2_ws/src/polyumi_ros2/test/test_camera_preproc.py`` loads it **by path**. Two copies of the
digests could drift apart silently and would then be asserting nothing; one copy cannot.

**Moving or renaming this file breaks the ROS-side test** — it hardcodes the path. Grep first.

Dependency-free beyond numpy, deliberately, so the ROS venv can execute it.
"""

import numpy as np

#: ``(height, width, sha256 hexdigest of resize_camera0_rgb(golden_frame(height, width)))``.
#:
#: The two shapes are the real ones: the Elgato inference capture, where the 4:3 crop is active,
#: and a ``gopro.mp4`` training frame, where it is a no-op. A changed digest means checkpoints
#: already trained are on a transform the inference node no longer reproduces — regenerate these
#: only when that is the intent, never to turn a red test green.
GOLDEN_VECTORS = [
    (1080, 1920, 'a9500a802012045302349f98f6390de6860bc74333765d1331715d4f89ea468f'),
    (2028, 2704, 'a4db72e016d56270136acb705af7d4d08d5899e9aeb0ae55c6b73576dc31a301'),
]


def golden_frame(h: int, w: int) -> np.ndarray:
    """
    Build a fixed RGB test image without an RNG.

    numpy guarantees nothing about ``default_rng``'s stream across versions, and the two
    environments this vector has to agree in are two numpy majors apart — seeding would let the
    digests diverge for a reason having nothing to do with the transform. Plain uint8 arithmetic
    (wrapping on overflow) is stable everywhere. Each channel gets a different pattern so a
    channel-order regression cannot slip through a digest that only checks the whole buffer.
    """
    x = (np.arange(w) % 256).astype(np.uint8)[None, :]
    y = (np.arange(h) % 256).astype(np.uint8)[:, None]
    frame = np.empty((h, w, 3), np.uint8)
    frame[..., 0] = x + y
    frame[..., 1] = x * 3
    frame[..., 2] = y * 5
    return frame

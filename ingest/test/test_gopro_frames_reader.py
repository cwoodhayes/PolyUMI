"""GoproMp4Frames decodes gopro.mp4 on demand with a zarr-Array-like surface."""

import pathlib

import cv2
import numpy as np
import pytest

from polyumi_ingest.video_helpers import GoproMp4Frames

_N = 12
_H, _W = 48, 64


def _write_test_mp4(path: pathlib.Path) -> int:
    """Write an _N-frame mp4 where frame i is a solid gray ramp; return frames written."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (_W, _H))
    if not writer.isOpened():
        return 0
    for i in range(_N):
        writer.write(np.full((_H, _W, 3), i * 20, dtype=np.uint8))
    writer.release()
    return _N


@pytest.fixture
def mp4_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a small decodable mp4, or skip if this OpenCV build can't encode one."""
    path = tmp_path / 'gopro.mp4'
    if _write_test_mp4(path) == 0 or not path.exists() or path.stat().st_size == 0:
        pytest.skip('cv2.VideoWriter (mp4v) unavailable in this environment')
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) >= _N
    cap.release()
    if not ok:
        pytest.skip('encoded mp4 not readable back in this environment')
    return path


def test_len_shape_dtype(mp4_path: pathlib.Path) -> None:
    """len/shape/dtype mirror the former gopro/frames array on the timestamps grid."""
    reader = GoproMp4Frames(mp4_path, _N)
    assert len(reader) == _N
    assert reader.shape == (_N, _H, _W, 3)
    assert reader.dtype == np.dtype('uint8')
    reader.close()


def test_sequential_frames_ascend(mp4_path: pathlib.Path) -> None:
    """Sequential forward reads return frames whose mean follows the encoded ramp."""
    reader = GoproMp4Frames(mp4_path, _N)
    means = [float(np.asarray(reader[i]).mean()) for i in range(_N)]
    reader.close()
    # Lossy codec, so allow slack, but the ramp must be monotonic overall.
    assert means[-1] > means[0] + 50
    assert np.all(np.diff(means) > -5)


def test_slice_returns_stack(mp4_path: pathlib.Path) -> None:
    """A contiguous slice returns a (k,H,W,3) stack (mcap batch access)."""
    reader = GoproMp4Frames(mp4_path, _N)
    batch = reader[3:7]
    reader.close()
    assert batch.shape == (4, _H, _W, 3)
    assert batch.dtype == np.uint8


def test_backward_seek_reopens(mp4_path: pathlib.Path) -> None:
    """Reading an earlier index after a later one transparently reopens and matches."""
    reader = GoproMp4Frames(mp4_path, _N)
    first_pass = np.asarray(reader[2]).copy()
    _ = reader[9]
    second_pass = np.asarray(reader[2])  # forces backward reopen
    reader.close()
    assert np.array_equal(first_pass, second_pass)


def test_out_of_range_raises(mp4_path: pathlib.Path) -> None:
    """Indexing at or beyond the pinned length raises IndexError."""
    reader = GoproMp4Frames(mp4_path, _N)
    with pytest.raises(IndexError):
        _ = reader[_N]
    reader.close()

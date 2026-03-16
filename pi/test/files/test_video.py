"""Unit tests for VideoFile."""

import csv

import cv2
import numpy as np
import pytest
from polyumi_pi.files.video import VideoFile

WIDTH = 64
HEIGHT = 48
FPS = 10.0


def make_jpeg(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """Return a minimal valid JPEG-encoded BGR frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode('.jpg', frame)
    assert ok
    return buf.tobytes()


@pytest.fixture
def video_file(tmp_path):
    """Return a VideoFile instance pointing at a temp directory."""
    return VideoFile(
        path=tmp_path / 'video.avi',
        fps=FPS,
        width=WIDTH,
        height=HEIGHT,
    )


# ---------------------------------------------------------------------------
# Construction / timestamps_path
# ---------------------------------------------------------------------------


def test_timestamps_path_default(tmp_path):
    """timestamps_path is derived from the AVI path when not supplied."""
    vf = VideoFile(path=tmp_path / 'video.avi', fps=FPS, width=WIDTH, height=HEIGHT)
    assert vf.timestamps_path == tmp_path / 'video_timestamps.csv'


def test_timestamps_path_explicit(tmp_path):
    """An explicit timestamps_path is preserved unchanged."""
    ts = tmp_path / 'custom_ts.csv'
    vf = VideoFile(
        path=tmp_path / 'video.avi',
        fps=FPS,
        width=WIDTH,
        height=HEIGHT,
        timestamps_path=ts,
    )
    assert vf.timestamps_path == ts


# ---------------------------------------------------------------------------
# recording() context manager
# ---------------------------------------------------------------------------


def test_recording_creates_files(video_file, tmp_path):
    """recording() creates both the AVI and the timestamps CSV."""
    with video_file.recording():
        pass

    assert (tmp_path / 'video.avi').is_file()
    assert (tmp_path / 'video_timestamps.csv').is_file()


def test_recording_cleans_up_state(video_file):
    """Internal writer/fp references are None after the context exits."""
    with video_file.recording():
        pass

    assert video_file._writer is None
    assert video_file._timestamps_fp is None
    assert video_file._timestamps_writer is None


def test_recording_reentrant_raises(video_file):
    """Opening a second recording() context while one is active raises."""
    with video_file.recording():
        with pytest.raises(RuntimeError, match='already active'):
            with video_file.recording():
                pass


# ---------------------------------------------------------------------------
# write_frame()
# ---------------------------------------------------------------------------


def test_write_frame_outside_context_raises(video_file):
    """write_frame() outside recording() raises RuntimeError."""
    with pytest.raises(RuntimeError, match='not active'):
        video_file.write_frame(make_jpeg())


def test_write_frame_increments_index(video_file):
    """Frame index increments with each write_frame() call."""
    with video_file.recording() as vf:
        assert vf._frame_idx == 0
        vf.write_frame(make_jpeg())
        assert vf._frame_idx == 1
        vf.write_frame(make_jpeg())
        assert vf._frame_idx == 2


def test_write_frame_timestamps_csv(video_file, tmp_path):
    """Each write_frame() appends one row with correct frame index to the CSV."""
    ts_values = [1_000_000_000, 2_000_000_000, 3_000_000_000]
    with video_file.recording() as vf:
        for ts in ts_values:
            vf.write_frame(make_jpeg(), timestamp_ns_value=ts)

    rows = list(csv.reader((tmp_path / 'video_timestamps.csv').open()))
    assert len(rows) == 3
    for i, (row, ts) in enumerate(zip(rows, ts_values)):
        assert int(row[0]) == i
        assert int(row[1]) == ts


def test_write_frame_uses_time_ns_when_no_timestamp(video_file):
    """write_frame() records a plausible ns timestamp when none is supplied."""
    from time import time_ns

    before = time_ns()
    with video_file.recording() as vf:
        vf.write_frame(make_jpeg())
    after = time_ns()

    rows = list(csv.reader(video_file.timestamps_path.open()))
    ts = int(rows[0][1])
    assert before <= ts <= after


def test_write_frame_wrong_size_raises(video_file):
    """write_frame() raises if the decoded JPEG has the wrong dimensions."""
    wrong_jpeg = make_jpeg(width=WIDTH * 2, height=HEIGHT * 2)
    with video_file.recording() as vf:
        with pytest.raises(ValueError, match='Frame size mismatch'):
            vf.write_frame(wrong_jpeg)


def test_write_frame_invalid_jpeg_raises(video_file):
    """write_frame() raises if the bytes cannot be decoded as a JPEG."""
    with video_file.recording() as vf:
        with pytest.raises(ValueError, match='Failed to decode'):
            vf.write_frame(b'not a jpeg')


# ---------------------------------------------------------------------------
# from_file() round-trip
# ---------------------------------------------------------------------------


def test_from_file_roundtrip(video_file, tmp_path):
    """from_file() recovers fps, width, and height written by recording()."""
    with video_file.recording() as vf:
        vf.write_frame(make_jpeg())

    loaded = VideoFile.from_file(tmp_path / 'video.avi')

    assert loaded.fps == pytest.approx(FPS, rel=0.01)
    assert loaded.width == WIDTH
    assert loaded.height == HEIGHT

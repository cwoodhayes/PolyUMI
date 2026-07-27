"""Tests for on-demand gopro.mp4 thumbnail decoding (Phase 4)."""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest
from polyumi_catalog import thumbnails

_N = 40
_H, _W = 48, 64


def _write_test_mp4(path: pathlib.Path, n_frames: int = _N) -> int:
    """Write an n_frames mp4 of solid-color frames; return frames actually written."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (_W, _H))
    if not writer.isOpened():
        return 0
    for i in range(n_frames):
        writer.write(np.full((_H, _W, 3), (i * 5) % 256, dtype=np.uint8))
    writer.release()
    return n_frames


@pytest.fixture
def session_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a session directory containing a small decodable gopro.mp4, or skip."""
    sd = tmp_path / 'session_1'
    sd.mkdir()
    path = sd / 'gopro.mp4'
    if _write_test_mp4(path) == 0 or not path.exists() or path.stat().st_size == 0:
        pytest.skip('cv2.VideoWriter (mp4v) unavailable in this environment')
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) >= _N
    cap.release()
    if not ok:
        pytest.skip('encoded mp4 not readable back in this environment')
    return sd


def test_session_thumbnail_jpeg_returns_none_without_gopro_mp4(tmp_path: pathlib.Path):
    """A session directory with no gopro.mp4 has no thumbnail, not an error."""
    sd = tmp_path / 'session_1'
    sd.mkdir()
    assert thumbnails.session_thumbnail_jpeg(sd) is None


def test_session_thumbnail_jpeg_decodes_a_valid_jpeg(session_dir: pathlib.Path):
    """A readable gopro.mp4 yields JPEG bytes that decode back to a plausible frame."""
    jpeg = thumbnails.session_thumbnail_jpeg(session_dir)
    assert jpeg is not None
    assert jpeg[:2] == b'\xff\xd8'  # JPEG magic

    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[0] > 0
    assert decoded.shape[1] <= thumbnails._THUMBNAIL_WIDTH


def test_session_thumbnail_jpeg_handles_short_video(tmp_path: pathlib.Path):
    """A video shorter than the target frame index still yields a thumbnail (last frame)."""
    sd = tmp_path / 'session_short'
    sd.mkdir()
    path = sd / 'gopro.mp4'
    if _write_test_mp4(path, n_frames=3) == 0 or not path.exists():
        pytest.skip('cv2.VideoWriter (mp4v) unavailable in this environment')
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 1
    cap.release()
    if not ok:
        pytest.skip('encoded mp4 not readable back in this environment')

    jpeg = thumbnails.session_thumbnail_jpeg(sd)
    assert jpeg is not None
    assert jpeg[:2] == b'\xff\xd8'


def test_session_thumbnail_jpeg_returns_none_for_corrupt_file(tmp_path: pathlib.Path):
    """A gopro.mp4 that isn't actually a video is treated as 'no thumbnail', not a crash."""
    sd = tmp_path / 'session_bad'
    sd.mkdir()
    (sd / 'gopro.mp4').write_bytes(b'not a real video file')
    assert thumbnails.session_thumbnail_jpeg(sd) is None


def test_gopro_fps_returns_none_without_gopro_mp4(tmp_path: pathlib.Path):
    """A session directory with no gopro.mp4 has no fps, not an error."""
    sd = tmp_path / 'session_1'
    sd.mkdir()
    assert thumbnails.gopro_fps(sd) is None


def test_gopro_fps_reads_native_rate(session_dir: pathlib.Path):
    """A readable gopro.mp4 reports the fps it was encoded at (_write_test_mp4 uses 30.0)."""
    fps = thumbnails.gopro_fps(session_dir)
    assert fps is not None
    assert fps == pytest.approx(30.0, abs=0.5)


def test_gopro_fps_returns_none_for_corrupt_file(tmp_path: pathlib.Path):
    """A gopro.mp4 that isn't actually a video is treated as 'fps unavailable', not a crash."""
    sd = tmp_path / 'session_bad'
    sd.mkdir()
    (sd / 'gopro.mp4').write_bytes(b'not a real video file')
    assert thumbnails.gopro_fps(sd) is None

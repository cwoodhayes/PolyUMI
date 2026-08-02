"""Tests for frame decoding into zarr, including the damaged-JPEG paths."""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import zarr
from polyumi_ingest.video_helpers import write_frames_to_zarr

_H, _W = 4, 6


def _write_jpegs(video_dir: pathlib.Path, n: int, empty: set[int] = frozenset()) -> list[pathlib.Path]:
    """Write ``n`` JPEGs, leaving the ones in ``empty`` zero-length as a killed recorder would."""
    video_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        path = video_dir / f'frame_{i:06d}.jpg'
        if i in empty:
            path.write_bytes(b'')
        else:
            frame = rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8)
            path.write_bytes(cv2.imencode('.jpg', frame)[1].tobytes())
        paths.append(path)
    return paths


def _frames_array(tmp_path: pathlib.Path, n: int) -> zarr.Array:
    root = zarr.open_group(str(tmp_path / 'out.zarr'), mode='w', zarr_format=2)
    return root.zeros(name='frames', shape=(n, _H, _W, 3), dtype='uint8', chunks=(1, _H, _W, 3))


def test_zero_byte_final_frame_truncates_instead_of_raising(tmp_path: pathlib.Path) -> None:
    """
    The reported bug, in miniature: a 0-byte last JPEG must not kill the episode.

    cv2.imdecode *raises* on an empty buffer rather than returning None, which is how a single
    truncated frame used to abort a whole scene's build.
    """
    paths = _write_jpegs(tmp_path / 'video', 3, empty={2})

    n_written = write_frames_to_zarr(paths, _frames_array(tmp_path, 3))

    assert n_written == 2


def test_corrupt_middle_frame_truncates_rather_than_skipping(tmp_path: pathlib.Path) -> None:
    """
    Decoding stops at the bad frame; later frames are dropped, not shifted up.

    Frame *i* lines up with row *i* of video_timestamps.csv, so silently skipping one would
    misalign every timestamp after it — worse than losing the tail.
    """
    paths = _write_jpegs(tmp_path / 'video', 5, empty={2})

    n_written = write_frames_to_zarr(paths, _frames_array(tmp_path, 5))

    assert n_written == 2


def test_all_frames_decode_when_none_are_damaged(tmp_path: pathlib.Path) -> None:
    """Sanity check that the guard doesn't truncate healthy input."""
    paths = _write_jpegs(tmp_path / 'video', 4)

    assert write_frames_to_zarr(paths, _frames_array(tmp_path, 4)) == 4

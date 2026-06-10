"""JPEG frame directory abstraction for PolyUMI data collection."""

from __future__ import annotations

import csv
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import time_ns
from typing import Any, Generator, TextIO

from polyumi_pi.files.base import SessionDataABC

_FRAME_GLOB = 'frame_*.jpg'
_FRAME_NAME = 'frame_{:06d}.jpg'
_TIMESTAMPS_NAME = 'video_timestamps.csv'


@dataclass
class VideoFile(SessionDataABC):
    """
    Abstraction for a directory of JPEG frames recorded during data collection.

    Each frame is written as a raw JPEG file (no decode/re-encode).
    A sidecar CSV maps frame index to nanosecond timestamp.
    """

    fps: float
    """Nominal capture framerate — stored in metadata, not enforced here."""

    width: int
    """Expected frame width in pixels. Validated on write."""

    height: int
    """Expected frame height in pixels. Validated on write."""

    _timestamps_fp: TextIO | None = field(init=False, default=None, repr=False)
    _timestamps_writer: Any | None = field(init=False, default=None, repr=False)
    _frame_idx: int = field(init=False, default=0, repr=False)

    @property
    def timestamps_path(self) -> pathlib.Path:
        """Path to the sidecar CSV file."""
        return self.path / _TIMESTAMPS_NAME

    @classmethod
    def from_file(cls, path: pathlib.Path) -> VideoFile:
        """
        Load video parameters from an existing frame directory.

        Reads fps, width, and height from the first frame found.
        """
        if not path.is_dir():
            raise ValueError(f'Expected frame directory, got: {path}')

        frames = sorted(path.glob(_FRAME_GLOB))
        if not frames:
            raise ValueError(f'No JPEG frames found in: {path}')

        # import here to avoid pulling cv2 into contexts where it may not
        # be installed (e.g. the PC-side fetch script before cv2 is set up)
        import cv2
        import numpy as np

        sample = cv2.imdecode(
            np.frombuffer(frames[0].read_bytes(), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if sample is None:
            raise ValueError(f'Failed to decode sample frame: {frames[0]}')

        height, width = sample.shape[:2]

        # fps is not stored in the frames themselves; try to recover from
        # the parent session metadata if available, otherwise default to 0.
        # Callers that need fps should pass it explicitly.
        return cls(path=path, fps=0.0, width=width, height=height)

    @contextmanager
    def recording(self) -> Generator[VideoFile, None, None]:
        """
        Context manager that opens the frame directory and sidecar CSV for writing.

        Frames should be written via :meth:`write_frame` while this context
        manager is active.
        """
        if self._timestamps_fp is not None:
            raise RuntimeError('Video recording already active for this file.')

        self.path.mkdir(parents=True, exist_ok=True)
        timestamps_fp = self.timestamps_path.open('w', newline='')
        timestamps_writer = csv.writer(timestamps_fp)

        self._timestamps_fp = timestamps_fp
        self._timestamps_writer = timestamps_writer
        self._frame_idx = 0

        try:
            yield self
        finally:
            timestamps_fp.close()
            self._timestamps_fp = None
            self._timestamps_writer = None

    def write_frame(
        self,
        jpg_frame: bytes,
        timestamp_ns_value: int | None = None,
    ) -> None:
        """Write one JPEG frame and append its timestamp to the sidecar CSV."""
        if self._timestamps_writer is None:
            raise RuntimeError('Video recording is not active. Use this inside recording().')

        ts_ns = time_ns() if timestamp_ns_value is None else timestamp_ns_value
        frame_path = self.path / _FRAME_NAME.format(self._frame_idx)
        frame_path.write_bytes(jpg_frame)
        self._timestamps_writer.writerow([self._frame_idx, ts_ns])

        timestamps_fp = self._timestamps_fp
        if timestamps_fp is not None:
            timestamps_fp.flush()

        self._frame_idx += 1

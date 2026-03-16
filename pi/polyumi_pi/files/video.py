"""AVI video file abstraction for PolyUMI data collection."""

from __future__ import annotations

import csv
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import time_ns
from typing import Any, Generator, TextIO

import cv2
import numpy as np

from polyumi_pi.files.base import SessionDataABC


@dataclass
class VideoFile(SessionDataABC):
    """Abstraction for an AVI video file recorded during data collection."""

    fps: float
    """Target framerate for this video file."""

    width: int
    """Frame width in pixels."""

    height: int
    """Frame height in pixels."""

    timestamps_path: pathlib.Path | None = None
    """Path to sidecar CSV with frame index and nanosecond timestamp."""

    _writer: cv2.VideoWriter | None = field(init=False, default=None, repr=False)
    _timestamps_fp: TextIO | None = field(init=False, default=None, repr=False)
    _timestamps_writer: Any | None = field(init=False, default=None, repr=False)
    _frame_idx: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.timestamps_path is None:
            self.timestamps_path = self.path.with_name(
                f'{self.path.stem}_timestamps.csv'
            )

    @classmethod
    def from_file(cls, path: pathlib.Path) -> VideoFile:
        """Load video parameters from an existing AVI file."""
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f'Failed to open video file: {path}')

        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()

        return cls(path=path, fps=fps, width=width, height=height)

    @contextmanager
    def recording(self) -> Generator[VideoFile, None, None]:
        """
        Context manager that opens an AVI file and sidecar timestamp CSV.

        Frames should be written via :meth:`write_frame` while this context
        manager is active.
        """
        if self._writer is not None:
            raise RuntimeError('Video recording already active for this file.')

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # type: ignore
        writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not writer.isOpened():
            raise ValueError(f'Failed to open video writer for path: {self.path}')

        timestamps_path = self.timestamps_path
        if timestamps_path is None:
            raise RuntimeError('timestamps_path must be set before recording.')

        timestamps_fp = timestamps_path.open('w', newline='')
        timestamps_writer = csv.writer(timestamps_fp)

        self._writer = writer
        self._timestamps_fp = timestamps_fp
        self._timestamps_writer = timestamps_writer
        self._frame_idx = 0

        try:
            yield self
        finally:
            writer.release()
            timestamps_fp.close()
            self._writer = None
            self._timestamps_fp = None
            self._timestamps_writer = None

    def write_frame(
        self, jpg_frame: bytes, timestamp_ns_value: int | None = None
    ) -> None:
        """Write one JPEG-encoded frame and append its timestamp to CSV."""
        if self._writer is None or self._timestamps_writer is None:
            raise RuntimeError(
                'Video recording is not active. Use this inside recording().'
            )

        frame = cv2.imdecode(
            np.frombuffer(jpg_frame, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            raise ValueError('Failed to decode JPEG frame bytes.')

        frame_height, frame_width = frame.shape[:2]
        if frame_width != self.width or frame_height != self.height:
            raise ValueError(
                'Frame size mismatch. '
                f'Expected ({self.width}, {self.height}), '
                f'got ({frame_width}, {frame_height}).'
            )

        ts_ns = time_ns() if timestamp_ns_value is None else timestamp_ns_value
        self._writer.write(frame)
        self._timestamps_writer.writerow([self._frame_idx, ts_ns])

        timestamps_fp = self._timestamps_fp
        if timestamps_fp is not None:
            timestamps_fp.flush()

        self._frame_idx += 1

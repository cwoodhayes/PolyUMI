"""ingest/video_helpers.py - Encode PolyUMI session data into MP4 files via ffmpeg."""

import logging
import os
import pathlib
import subprocess
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
import zarr
from polyumi_pi.files.session import SessionFiles

log = logging.getLogger(__name__)


class GoproMp4Frames:
    """
    Sequential-access, zarr-Array-like view over a gopro.mp4's decoded frames.

    Presents the subset of the zarr ``Array`` surface the frame consumers use
    (``__len__``, ``.shape``, ``.dtype``, integer and slice ``__getitem__``,
    returning ``(H,W,3)`` / ``(k,H,W,3)`` uint8 **RGB**), decoding on demand from
    the mp4 rather than a stored ``gopro/frames`` array. This lets the pipeline
    drop the ~70x-larger JpegXl re-encode of frames the mp4 already holds.

    ``len`` is pinned to the authoritative ``timestamps/gopro`` grid length so an
    index maps 1:1 onto the former zarr frame index. Access is intended to be
    forward/sequential — every consumer iterates a contiguous ascending range —
    so decode is a single forward pass, skipping to a target with ``cap.grab()``.
    A backward request transparently reopens from the start (rare).

    NOT thread-safe: a single OpenCV ``VideoCapture`` backs it, so never index it
    from multiple threads. Call ``close()`` (or let it be garbage-collected) to
    release the capture.
    """

    def __init__(self, path: pathlib.Path, n_frames: int) -> None:
        """Bind the reader to a gopro.mp4 at ``path`` with a pinned ``n_frames`` length."""
        self._path = pathlib.Path(path)
        self._n = int(n_frames)
        self._cap: cv2.VideoCapture | None = None
        self._next = 0  # index the next cap.read() will return
        self._h = 0
        self._w = 0

    def _ensure_open(self) -> cv2.VideoCapture:
        if self._cap is None:
            cap = cv2.VideoCapture(str(self._path))
            if not cap.isOpened():
                raise RuntimeError(f'Could not open GoPro video: {self._path}')
            self._cap = cap
            self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._next = 0
        return self._cap

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return (n_frames, H, W, 3), matching the former gopro/frames array."""
        self._ensure_open()
        return (self._n, self._h, self._w, 3)

    @property
    def dtype(self) -> np.dtype:
        """Return the frame dtype (uint8), matching the former array."""
        return np.dtype('uint8')

    def __len__(self) -> int:
        return self._n

    def close(self) -> None:
        """Release the underlying VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:  # noqa: D105
        self.close()

    def _read_index(self, i: int) -> np.ndarray:
        if not (0 <= i < self._n):
            raise IndexError(f'frame index {i} out of range [0, {self._n})')
        cap = self._ensure_open()
        if i < self._next:
            # Backward seek: reopen from the start. Consumers go forward, so this is rare.
            cap.release()
            cap = cv2.VideoCapture(str(self._path))
            if not cap.isOpened():
                raise RuntimeError(f'Could not reopen GoPro video: {self._path}')
            self._cap = cap
            self._next = 0
        while self._next < i:
            if not cap.grab():
                raise RuntimeError(f'{self._path}: unexpected EOF skipping to frame {i} (at {self._next})')
            self._next += 1
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f'{self._path}: failed to read frame {i}')
        self._next += 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, key: int | np.integer | slice) -> np.ndarray:
        if isinstance(key, slice):
            indices = range(*key.indices(self._n))
            if not indices:
                return np.empty((0, self._h, self._w, 3), dtype=np.uint8)
            return np.stack([self._read_index(i) for i in indices], axis=0)
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i < 0:
                i += self._n
            return self._read_index(i)
        raise TypeError(f'GoproMp4Frames index must be int or slice, got {type(key)!r}')


def open_gopro_frames(ep_grp: zarr.Group, scene_zarr: pathlib.Path) -> GoproMp4Frames:
    """
    Return a GoproMp4Frames reader for an episode, resolving its gopro.mp4 sidecar.

    The reader length is pinned to the episode's ``timestamps/gopro`` grid, so it
    is a drop-in for the former ``gopro/frames`` array on that same index grid.
    """
    # Imported lazily: scene_files pulls in the pzarr package, whose __init__ imports
    # store, which imports this module — a top-level import here would be circular.
    from polyumi_ingest.pzarr.scene_files import resolve_gopro_mp4

    mp4 = resolve_gopro_mp4(ep_grp, scene_zarr)
    n_frames = int(ep_grp['timestamps/gopro'].shape[0])  # type: ignore[union-attr]
    return GoproMp4Frames(mp4, n_frames)


def _video_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _image_frames(paths: list[pathlib.Path]) -> Iterator[np.ndarray]:
    for i, fp in enumerate(paths):
        raw = np.frombuffer(fp.read_bytes(), dtype=np.uint8)
        # A recorder killed mid-write leaves a zero-byte or half-flushed JPEG behind, in
        # practice as the final frame. cv2.imdecode *raises* on an empty buffer and returns
        # None on a corrupt one, so guard both. Stop rather than skip: frame i lines up with
        # row i of video_timestamps.csv, so dropping one from the middle would silently
        # shift every timestamp after it. The caller truncates to what decoded.
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
        if bgr is None:
            log.warning(f'  Undecodable frame {fp.name} ({raw.size} bytes); truncating after {i} of {len(paths)}.')
            return
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_frames_to_zarr(
    source: pathlib.Path | list[pathlib.Path],
    frames_arr: zarr.Array,
    *,
    num_workers: int | None = None,
) -> int:
    """
    Decode source and write each frame into the pre-created frames_arr.

    source may be a video file path (decoded with cv2.VideoCapture) or a list
    of image file paths (decoded individually with cv2.imdecode).

    Producer-consumer: the calling thread decodes frames sequentially
    (VideoCapture is not thread-safe); a pool of workers compresses and writes
    chunks concurrently (zarr DirectoryStore is safe for concurrent chunk
    writes). A semaphore bounds the number of decoded frames held in memory at
    once — important for high-resolution footage.

    Returns the number of frames actually written.
    """
    n_workers = num_workers if num_workers is not None else (os.cpu_count() or 1)
    # each decoded 4K frame ≈ 25 MB; cap in-flight to avoid unbounded buffering
    in_flight = threading.Semaphore(n_workers * 2)

    cap = None
    if isinstance(source, list):
        source_desc = f'{len(source)} image files'
        frame_iter = _image_frames(source)
    else:
        source_desc = source.name
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f'Could not open video: {source}')
        frame_iter = _video_frames(cap)

    futures: list[Future[None]] = []
    log.info(f'  Decoding {source_desc} with {n_workers} workers...')
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for j, frame in enumerate(frame_iter):
                in_flight.acquire()

                def _write(idx: int = j, f: np.ndarray = frame) -> None:
                    try:
                        frames_arr[idx] = f
                    finally:
                        in_flight.release()

                futures.append(pool.submit(_write))
        # ThreadPoolExecutor.__exit__ waits for all submitted tasks to finish
    finally:
        if cap is not None:
            cap.release()

    for fut in futures:
        fut.result()  # re-raise any worker exceptions

    elapsed = time.perf_counter() - t0
    n_written = len(futures)
    _, H, W, _ = frames_arr.shape
    uncompressed_mb = n_written * H * W * 3 / 1e6

    if n_written > 0 and elapsed > 0:
        log.info(
            f'  {n_written} frames in {elapsed:.1f}s'
            f' ({n_written / elapsed:.1f} fps, {uncompressed_mb / elapsed:.0f} MB/s uncompressed)'
        )
    else:
        log.warning(f'  No frames written from {source_desc}.')

    return n_written


def encode_session_video(
    session_path: pathlib.Path,
    fps: float,
    output_name: str,
    include_audio: bool,
) -> None:
    """Encode JPEG frames in a session directory into an MP4."""
    session_path = session_path.resolve()
    if not session_path.is_dir():
        raise RuntimeError(f'Session directory not found: {session_path}')

    video_dir = session_path / 'video'
    if not video_dir.is_dir():
        raise RuntimeError(f'No video directory found at {video_dir}')

    # prefer fps from session metadata if available
    try:
        session = SessionFiles.from_file(session_path)
        if session.metadata.camera_fps is not None:
            fps = float(session.metadata.camera_fps)
            log.info(f'Using fps from metadata for {session_path.name}: {fps}')
    except Exception as e:
        log.warning(f'Could not load metadata for {session_path.name}: {e}. Using --fps={fps}.')

    output_path = session_path / output_name
    audio_path = session_path / 'audio.wav'
    has_audio = include_audio and audio_path.is_file()

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate',
        str(fps),
        '-i',
        str(video_dir / 'frame_%06d.jpg'),
    ]

    if has_audio:
        cmd += ['-i', str(audio_path)]

    cmd += [
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',  # broadest playback compatibility
    ]

    if has_audio:
        cmd += ['-c:a', 'aac']

    cmd.append(str(output_path))

    log.info(f'Encoding: {" ".join(cmd)}')
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg exited with code {result.returncode}')

    log.info(f'Video written to {output_path}')

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


def _video_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _image_frames(paths: list[pathlib.Path]) -> Iterator[np.ndarray]:
    for fp in paths:
        raw = np.frombuffer(fp.read_bytes(), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'Failed to decode frame: {fp}')
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

"""ingest/video_helpers.py - Encode PolyUMI session data into MP4 files via ffmpeg."""

import logging
import os
import pathlib
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
import zarr
from polyumi_pi.files.session import SessionFiles

log = logging.getLogger(__name__)


def write_video_frames_to_zarr(
    video_path: pathlib.Path,
    frames_arr: zarr.Array,
    *,
    num_workers: int | None = None,
) -> int:
    """
    Decode video_path and write each frame into the pre-created frames_arr.

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

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    futures: list[Future[None]] = []

    log.info(f'  Decoding {video_path.name} with {n_workers} workers...')
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            j = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frame: np.ndarray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                in_flight.acquire()

                def _write(idx: int = j, f: np.ndarray = frame) -> None:
                    try:
                        frames_arr[idx] = f
                    finally:
                        in_flight.release()

                futures.append(pool.submit(_write))
                j += 1
        # ThreadPoolExecutor.__exit__ waits for all submitted tasks to finish
    finally:
        cap.release()

    for fut in futures:
        fut.result()  # re-raise any worker exceptions

    elapsed = time.perf_counter() - t0
    _, H, W, _ = frames_arr.shape
    uncompressed_mb = j * H * W * 3 / 1e6
    log.info(
        f'  {j} frames in {elapsed:.1f}s'
        f' ({j / elapsed:.1f} fps, {uncompressed_mb / elapsed:.0f} MB/s uncompressed)'
    )

    return j


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

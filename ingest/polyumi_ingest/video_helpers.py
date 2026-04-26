"""ingest/video_helpers.py - Encode PolyUMI session data into MP4 files via ffmpeg."""

import logging
import pathlib
import subprocess

from polyumi_pi.files.session import SessionFiles

log = logging.getLogger(__name__)


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

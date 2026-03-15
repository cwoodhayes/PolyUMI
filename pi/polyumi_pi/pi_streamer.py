"""
pi_streamer.py - Runs on the Raspberry Pi Zero 2W.

Streams MJPEG frames over ZMQ to pi_receiver_node on the host PC.

Usage:
    python pi_streamer.py stream
    python pi_streamer.py stream --port 5555 --width 640 --height 480 --fps 10
"""

import logging
import multiprocessing
import os

import typer
import zmq
from audio_streamer import AudioStreamer
from cam_streamer import CameraStreamer
from led_manager import LEDManager
from libcamera import controls  # type: ignore
from picamera2 import Picamera2
from polyumi_pi_msgs import camera_frame_pb2

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
log = logging.getLogger('pi_streamer')

app = typer.Typer()


def _stop_child_process(process: multiprocessing.Process | None) -> None:
    if process is None or not process.is_alive():
        return

    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        log.warning(f'Force-killing process pid={process.pid}')
        process.kill()
        process.join(timeout=2)


def _run_video_streamer(port: int, fps: int):
    context = zmq.Context()
    streamer = CameraStreamer(port=port, fps=fps, zmq_context=context)
    try:
        streamer.start()
    finally:
        context.term()


def _run_audio_streamer(
    port: int,
    sample_rate: int,
    chunk_ms: int,
    channels: int,
):
    context = zmq.Context()
    streamer = AudioStreamer(
        port=port,
        sample_rate=sample_rate,
        zmq_context=context,
        chunk_ms=chunk_ms,
        channels=channels,
    )
    try:
        streamer.start()
    finally:
        context.term()


@app.command()
def info():
    """Print camera information."""
    log.info(CameraStreamer.info())


@app.command()
def stream_video(
    port: int = typer.Option(5555, help='ZMQ PUSH port to bind on.'),
    fps: int = typer.Option(10, min=1, help='Target capture framerate (Hz).'),
):
    """Stream MJPEG frames over ZMQ."""
    log.info(f'Log level: {logging.getLevelName(log.level)}')
    context = zmq.Context()
    streamer = CameraStreamer(port=port, fps=fps, zmq_context=context)
    led = LEDManager()

    try:
        led.set_brightness(1.0)
        streamer.start()
    finally:
        context.term()
        led.set_brightness(0.0)


@app.command()
def stream_audio(
    port: int = typer.Option(5556, help='ZMQ PUSH port to bind on.'),
    sample_rate: int = typer.Option(44100, help='Audio sample rate (Hz).'),
    chunk_ms: int = typer.Option(20, help='Audio chunk size (ms).'),
    channels: int = typer.Option(2, help='Number of audio channels.'),
):
    """Stream audio data over ZMQ."""
    log.info(f'Log level: {logging.getLevelName(log.level)}')
    context = zmq.Context()
    streamer = AudioStreamer(
        port=port,
        sample_rate=sample_rate,
        zmq_context=context,
        chunk_ms=chunk_ms,
        channels=channels,
    )
    try:
        log.info('Starting audio streamer...')
        streamer.start()
    finally:
        context.term()


@app.command()
def stream(
    video_port: int = typer.Option(5555, help='ZMQ PUSH port for video.'),
    audio_port: int = typer.Option(5556, help='ZMQ PUSH port for audio.'),
    fps: int = typer.Option(10, min=1, help='Target capture framerate (Hz).'),
    sample_rate: int = typer.Option(16000, help='Audio sample rate (Hz).'),
    chunk_ms: int = typer.Option(20, help='Audio chunk size (ms).'),
    channels: int = typer.Option(1, help='Number of audio channels.'),
):
    """
    Stream both video and audio data over ZMQ.

    Intended for use on arm EE during inference.
    """
    log.info(f'Log level: {logging.getLevelName(log.level)}')
    led = LEDManager()
    cam_process: multiprocessing.Process | None = None
    audio_process: multiprocessing.Process | None = None

    try:
        led.set_brightness(1.0)
        log.info('Starting camera streamer...')
        cam_process = multiprocessing.Process(
            target=_run_video_streamer,
            args=(video_port, fps),
        )
        cam_process.start()

        log.info('Starting audio streamer...')
        audio_process = multiprocessing.Process(
            target=_run_audio_streamer,
            args=(audio_port, sample_rate, chunk_ms, channels),
        )
        audio_process.start()

        cam_process.join()
        audio_process.join()
    except KeyboardInterrupt:
        log.info('Keyboard interrupt received, stopping child streamers...')
    finally:
        _stop_child_process(cam_process)
        _stop_child_process(audio_process)
        led.set_brightness(0.0)


@app.command()
def record_episode(
    fps: int = typer.Option(10, min=1, help='Target capture framerate (Hz).'),
    sample_rate: int = typer.Option(16000, help='Audio sample rate (Hz).'),
    chunk_ms: int = typer.Option(20, help='Audio chunk size (ms).'),
    channels: int = typer.Option(1, help='Number of audio channels.'),
):
    """
    Record an episode; video and audio data is routed to local files.

    Intended for use on PolyUMI gripper during data recording.
    """
    log.info('Record command not implemented yet.')


if __name__ == '__main__':
    app()

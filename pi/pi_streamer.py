"""
pi_streamer.py - Runs on the Raspberry Pi Zero 2W.

Streams MJPEG frames over ZMQ to pi_receiver_node on the host PC.

Usage:
    python pi_streamer.py stream
    python pi_streamer.py stream --port 5555 --width 640 --height 480 --fps 10
"""

import io
import json
import logging
import os
import time

import numpy as np
import typer
import zmq
from cam_streamer import CameraStreamer
from led_manager import LEDManager
from libcamera import controls  # type: ignore
from picamera2 import Picamera2
from polyumi_pi_msgs import camera_frame_pb2

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
log = logging.getLogger('pi_streamer')

app = typer.Typer()


@app.command()
def info():
    """Print camera information."""
    log.info(CameraStreamer.info())


@app.command()
def stream(
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


if __name__ == '__main__':
    app()

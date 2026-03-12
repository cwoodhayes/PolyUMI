"""
pi_streamer.py - Runs on the Raspberry Pi Zero 2W.

Streams MJPEG frames over ZMQ to pi_receiver_node on the host PC.

Usage:
    python pi_streamer.py stream
    python pi_streamer.py stream --port 5555 --width 640 --height 480 --fps 10
"""

import logging
import time

import typer
import zmq
from picamera2 import Picamera2
from polyumi_pi_msgs import camera_frame_pb2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('pi_streamer')

app = typer.Typer()


@app.command()
def stream(
    port: int = typer.Option(5555, help='ZMQ PUSH port to bind on.'),
    width: int = typer.Option(640, help='Capture width in pixels.'),
    height: int = typer.Option(480, help='Capture height in pixels.'),
    fps: int = typer.Option(10, help='Target capture framerate (Hz).'),
):
    """Stream MJPEG frames over ZMQ."""
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind(f'tcp://*:{port}')
    log.info(f'ZMQ PUSH bound on tcp://*:{port}')

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={'size': (width, height), 'format': 'RGB888'},
    )
    cam.configure(config)
    cam.start()
    log.info(f'Camera started at {width}x{height} @ {fps}Hz')
    log.info(f'Publishing to tcp://<pi_ip>:{port}')

    interval = 1.0 / fps
    try:
        while True:
            t_start = time.monotonic()
            t_ns = time.time_ns()

            # Capture and encode frame as JPEG
            jpeg_bytes = cam.capture_image('main').tobytes()

            msg = camera_frame_pb2.CameraFrame()
            msg.timestamp_ns = t_ns
            msg.jpeg_data = jpeg_bytes
            msg.width = width
            msg.height = height
            socket.send(msg.SerializeToString(), zmq.NOBLOCK)

            elapsed = time.monotonic() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log.info('Interrupted, shutting down.')
    finally:
        cam.stop()
        socket.close()
        context.term()


if __name__ == '__main__':
    app()

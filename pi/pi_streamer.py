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
from io import BytesIO

import numpy as np
import typer
import zmq
from picamera2 import Picamera2
from polyumi_pi_msgs import camera_frame_pb2

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
log = logging.getLogger('pi_streamer')

app = typer.Typer()


@app.command()
def info():
    """Print camera information."""
    cam = Picamera2()
    controls = cam.camera_controls
    controls = json.dumps(controls, indent=2, default=str)
    log.info(f'Camera controls: {controls}\n\n\n')

    info = cam.sensor_modes
    info = json.dumps(info, indent=2, default=str)
    log.info(f'Camera sensor modes: {info}')


def compute_scaler_crop(
    cam: Picamera2,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Compute a top-aligned ScalerCrop rect for the requested aspect."""
    bounds = cam.camera_controls.get('ScalerCrop')
    rects: list[tuple[int, int, int, int]] = []
    if isinstance(bounds, tuple):
        for item in bounds:
            if isinstance(item, tuple) and len(item) == 4:
                rects.append(
                    (
                        int(item[0]),
                        int(item[1]),
                        int(item[2]),
                        int(item[3]),
                    )
                )

    if rects:
        base_x, base_y, sensor_width, sensor_height = max(
            rects,
            key=lambda rect: rect[2] * rect[3],
        )
    else:
        base_x = 0
        base_y = 0
        sensor_width, sensor_height = cam.sensor_resolution

    target_aspect = width / height
    sensor_aspect = sensor_width / sensor_height

    if target_aspect > sensor_aspect:
        crop_width = sensor_width
        crop_height = int(round(crop_width / target_aspect))
    else:
        crop_height = sensor_height
        crop_width = int(round(crop_height * target_aspect))

    x = base_x + max(0, (sensor_width - crop_width) // 2)
    y = base_y
    return (x, y, crop_width, crop_height)


@app.command()
def stream(
    port: int = typer.Option(5555, help='ZMQ PUSH port to bind on.'),
    width: int = typer.Option(640, help='Capture width in pixels.'),
    height: int = typer.Option(480, help='Capture height in pixels.'),
    fps: int = typer.Option(10, min=1, help='Target capture framerate (Hz).'),
):
    """Stream MJPEG frames over ZMQ."""
    log.info(f'Log level: {logging.getLevelName(log.level)}')
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f'tcp://*:{port}')
    log.info(f'ZMQ PUSH bound on tcp://*:{port}')

    # cam = Picamera2()
    # config = cam.create_video_configuration(
    #     main={'format': 'YUV420', 'size': (1600, 900)},
    # )
    # cam.configure(config)
    # cam.start()
    # scaler_crop = compute_scaler_crop(cam, width=width, height=height)
    # log.info(f'Camera started at {width}x{height} @ {fps}Hz')
    # log.info(f'Publishing to tcp://<pi_ip>:{port}')
    # cam.set_controls({'ScalerCrop': scaler_crop})
    # log.info(
    #     f'Requested ScalerCrop={scaler_crop}, '
    #     f'sensor={cam.sensor_resolution}, '
    #     f'control_bounds={cam.camera_controls.get("ScalerCrop")}'
    # )
    cam = Picamera2()
    # we want the 2nd mode for full FOV.
    mode = cam.sensor_modes[1]
    config = cam.create_video_configuration(
        main={'size': (2304 // 2, 1296 // 2), 'format': 'YUV420'},
        sensor={'output_size': mode['size'], 'bit_depth': mode['bit_depth']},
    )
    cam.configure(config)
    cam.start()

    interval = 1.0 / fps
    first_frame_logged = False
    try:
        while True:
            t_start = time.monotonic()

            # Capture and encode frame as JPEG
            data = io.BytesIO()
            cam.capture_file(data, format='jpeg')
            metadata = cam.capture_metadata()
            if not first_frame_logged:
                log.info(f'First-frame metadata: {metadata}')
                first_frame_logged = True
            log.debug(metadata)

            msg = camera_frame_pb2.CameraFrame()
            msg.timestamp_ns = metadata['SensorTimestamp']
            msg.jpeg_data = data.getvalue()
            msg.width = width
            msg.height = height
            try:
                socket.send(msg.SerializeToString(), zmq.NOBLOCK)
            except zmq.Again:
                log.debug('Dropped frame: receiver not ready.')

            log.debug(
                f'Captured frame at {msg.timestamp_ns} ns, '
                f'size={len(msg.jpeg_data)} bytes'
            )

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

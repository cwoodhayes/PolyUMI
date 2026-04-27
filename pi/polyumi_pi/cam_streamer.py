"""Code for streaming camera data."""

import contextlib
import io
import json
import logging
import signal
import time
from multiprocessing.connection import Connection

import zmq
from libcamera import controls  # type: ignore
from picamera2 import Picamera2
from polyumi_pi_msgs import camera_frame_pb2

from polyumi_pi.files.session import SessionFiles

log = logging.getLogger('pi_cam_process')


class CameraStreamer:
    """Class for streaming camera data."""

    VIEW_WIDTH = 620
    VIEW_HEIGHT = 480
    CAPTURE_WIDTH = 2304 // 2
    CAPTURE_HEIGHT = 1296 // 2

    def __init__(
        self,
        port: int | None,
        fps: int,
        zmq_context: zmq.Context,
        session: SessionFiles | None = None,
        stats_conn: Connection | None = None,
    ):
        """Initialize the camera streamer."""
        self.port = port
        self.fps = fps
        self.zmq_context = zmq_context
        self.session = session
        self.stats_conn = stats_conn
        self.cam = Picamera2()

    @classmethod
    def info(cls) -> str:
        """Return a camera information string."""
        cam = Picamera2()
        msg = []
        controls = json.dumps(cam.camera_controls, indent=2, default=str)
        msg.append(f'Camera controls: {controls}')

        info = json.dumps(cam.sensor_modes, indent=2, default=str)
        msg.append(f'Camera sensor modes: {info}')

        return '\n\n\n'.join(msg)

    def start(self) -> None:
        """Start streaming camera data."""
        streaming_enabled = self.port is not None
        socket = None
        if streaming_enabled:
            socket = self.zmq_context.socket(zmq.PUSH)
            socket.setsockopt(zmq.SNDHWM, 2)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(f'tcp://*:{self.port}')
            log.info(f'ZMQ PUSH bound on tcp://*:{self.port}')
        else:
            log.info('ZMQ video streaming disabled (port is None).')

        self.configure_camera()
        self.cam.start()
        self.set_initial_controls()

        log.info(f'Publishing to tcp://<pi_ip>:{self.port}')

        if self.session is not None and self.session.video is not None:
            log.info(f'Video will be recorded to {self.session.video.path}')

        interval = 1.0 / self.fps
        first_frame_metadata: dict | None = None
        stop_requested = False
        n_video_frames = 0
        n_video_dropped_frames = 0

        def handle_shutdown(signum, _frame):
            nonlocal stop_requested
            log.info(f'Received {signal.Signals(signum).name}. shutting down.')
            stop_requested = True

        prev_sigint = signal.signal(signal.SIGINT, handle_shutdown)
        prev_sigterm = signal.signal(signal.SIGTERM, handle_shutdown)

        with contextlib.ExitStack() as stack:
            if self.session is not None and self.session.video is not None:
                video_recorder = stack.enter_context(self.session.video.recording())
            else:
                video_recorder = None

            try:
                while not stop_requested:
                    t_start = time.monotonic()

                    # Capture and encode frame as JPEG
                    data = io.BytesIO()
                    self.cam.capture_file(data, format='jpeg')
                    metadata = self.cam.capture_metadata()
                    if first_frame_metadata is None:
                        first_frame_metadata = dict(metadata)
                        log.info(f'First-frame metadata: {first_frame_metadata}')
                    log.debug(metadata)

                    msg = camera_frame_pb2.CameraFrame()
                    msg.timestamp_ns = metadata['SensorTimestamp']
                    msg.jpeg_data = data.getvalue()
                    msg.width = self.VIEW_WIDTH
                    msg.height = self.VIEW_HEIGHT
                    if socket is not None:
                        try:
                            socket.send(msg.SerializeToString(), zmq.NOBLOCK)
                        except zmq.Again:
                            log.debug('Dropped frame: receiver not ready.')
                            n_video_dropped_frames += 1

                    if video_recorder is not None:
                        video_recorder.write_frame(data.getvalue(), msg.timestamp_ns)
                        n_video_frames += 1

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
                signal.signal(signal.SIGINT, prev_sigint)
                signal.signal(signal.SIGTERM, prev_sigterm)
                self.cam.stop()
                if socket is not None:
                    socket.close()

        if self.stats_conn is not None:
            try:
                self.stats_conn.send(
                    {
                        'n_video_frames': n_video_frames,
                        'video_dropped_frames': n_video_dropped_frames,
                        'first_frame_metadata': first_frame_metadata,
                    }
                )
            finally:
                self.stats_conn.close()

    def compute_scaler_crop(
        self,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        """Compute a top-right ScalerCrop rect for the requested aspect."""
        bounds = self.cam.camera_controls.get('ScalerCrop')
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
            sensor_width, sensor_height = self.cam.sensor_resolution

        target_aspect = width / height
        sensor_aspect = sensor_width / sensor_height

        if target_aspect > sensor_aspect:
            crop_width = sensor_width
            crop_height = int(round(crop_width / target_aspect))
        else:
            crop_height = sensor_height
            crop_width = int(round(crop_height * target_aspect))

        x = base_x + max(0, sensor_width - crop_width)
        y = base_y
        return (x, y, crop_width, crop_height)

    def configure_camera(self) -> None:
        """Configure the camera for our specific use-case."""
        # we want the 2nd mode for full FOV.
        mode = self.cam.sensor_modes[1]
        config = self.cam.create_video_configuration(
            main={'size': (2304 // 2, 1296 // 2), 'format': 'YUV420'},
            sensor={
                'output_size': mode['size'],
                'bit_depth': mode['bit_depth'],
            },
        )
        self.cam.configure(config)

    def set_initial_controls(self) -> None:
        """Set initial camera controls for our use-case."""
        scaler_crop = self.compute_scaler_crop(
            width=self.VIEW_WIDTH, height=self.VIEW_HEIGHT
        )
        # empirically determined in m
        dist_to_sensor = 0.2
        self.cam.set_controls(
            {
                'ScalerCrop': scaler_crop,
                'AfMode': controls.AfModeEnum.Manual,
                'LensPosition': 1.0 / dist_to_sensor,
            }
        )
        log.info(
            f'Requested ScalerCrop={scaler_crop}, '
            f'sensor={self.cam.sensor_resolution}, '
            f'control_bounds={self.cam.camera_controls.get("ScalerCrop")}'
        )

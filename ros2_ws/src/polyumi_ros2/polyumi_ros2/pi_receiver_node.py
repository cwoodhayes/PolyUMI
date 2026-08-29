"""
pi_receiver_node.py - ROS2 node running on the host PC.

Receives MJPEG frames from pi_streamer.py over ZMQ and publishes them
as sensor_msgs/CompressedImage on /pi/camera/image/compressed.

Dependencies:
    pip install pyzmq protobuf
    ROS: rclpy sensor_msgs

Usage:
    ros2 run polyumi_ros2 pi_receiver_node
    ros2 run polyumi_ros2 pi_receiver_node --ros-args \
        -p pi_host:=polyumi-pi.local -p port:=5555
"""

import logging
import threading

import rclpy
import rclpy.time
import zmq
from builtin_interfaces.msg import Time
from foxglove_msgs.msg import RawAudio
from polyumi_pi_msgs import audio_chunk_pb2, camera_frame_pb2
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('pi_receiver_node')

# ZMQ recv timeout (ms). On timeout we log a throttled "waiting for the Pi"
# warning instead of blocking forever — the most common reason no /pi/* messages
# appear is that the Pi stream (`polyumi-pi stream`) simply isn't running.
ZMQ_RECV_TIMEOUT_MS = 1000

# Minimum interval between repeats of the same warning (nanoseconds). Everything throttled here
# says "the Pi link is wrong"; once a second is plenty.
WARN_INTERVAL_NS = 1_000_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class LogThrottle:
    """Rate-limits repeated log lines, independently per key."""

    def __init__(self, interval_ns: int) -> None:
        """Initialize with the minimum interval between repeats of any one key."""
        self._interval_ns = interval_ns
        self._last: dict[str, int] = {}

    def should_emit(self, key: str, now_ns: int) -> bool:
        """Return True if ``key`` has not fired within the interval, recording this firing."""
        last = self._last.get(key)
        if last is not None and now_ns - last < self._interval_ns:
            return False
        self._last[key] = now_ns
        return True


def stamp_is_skewed(now_ns: int, stamp_ns: int, max_skew_s: float) -> bool:
    """
    Return True when a Pi stamp sits too far from this host's clock to be transport delay.

    Symmetric: a Pi clock running ahead is as broken as one running behind.
    """
    return abs(now_ns - stamp_ns) / 1e9 > max_skew_s


def ns_to_ros_time(t_ns: int) -> Time:
    """
    Re-split an epoch-nanosecond capture instant into a ROS2 Time message.

    The Pi already stamps both streams in this domain (the ``timestamp_ns`` contract in
    ``camera_frame.proto`` / ``audio_chunk.proto``), so this only re-splits — it never adjusts.
    """
    msg = Time()
    msg.sec = t_ns // 1_000_000_000
    msg.nanosec = t_ns % 1_000_000_000
    return msg


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------


class PiReceiverNode(Node):
    """Receive MJPEG frames over ZMQ and publish ROS2 compressed images."""

    def __init__(self):
        """Initialize ROS publishers, parameters, and receive thread."""
        super().__init__('pi_receiver_node')

        self.declare_parameter('pi_host', '10.106.10.62')
        self.declare_parameter('port', 5555)
        self.declare_parameter('audio_port', 5556)
        # |now - stamp| above which the Pi link is reported as mis-clocked. Well above the real
        # transport delay (tens of ms) and well below the failures worth catching: an
        # unsynchronised Pi clock, or a stamp that never left its device time base at all.
        self.declare_parameter('max_clock_skew_s', 0.5)

        self._pi_host = self.get_parameter('pi_host').get_parameter_value().string_value
        self._port = self.get_parameter('port').get_parameter_value().integer_value
        self._audio_port = self.get_parameter('audio_port').get_parameter_value().integer_value
        self._max_clock_skew_s = self.get_parameter('max_clock_skew_s').get_parameter_value().double_value

        self.camera_pub = self.create_publisher(
            CompressedImage,
            'camera/image/compressed',
            qos_profile=10,
        )
        self.audio_pub = self.create_publisher(
            RawAudio,
            'audio/raw',
            qos_profile=10,
        )

        self._zmq_context = zmq.Context()

        self._throttle = LogThrottle(WARN_INTERVAL_NS)

        recv_thread = threading.Thread(target=self._camera_recv_loop, daemon=True)
        recv_thread.start()

        audio_recv_thread = threading.Thread(target=self._audio_recv_loop, daemon=True)
        audio_recv_thread.start()

        self.get_logger().info(
            f'Receiving from tcp://{self._pi_host}:{self._port}, publishing on /pi/camera/image/compressed'
        )
        self.get_logger().info(f'Receiving audio from tcp://{self._pi_host}:{self._audio_port}')
        self.get_logger().info('Publishing audio on /pi/audio/raw')

    def _camera_recv_loop(self):
        sock = self._zmq_context.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT_MS)
        sock.connect(f'tcp://{self._pi_host}:{self._port}')

        while rclpy.ok():
            try:
                raw = sock.recv()
            except zmq.Again:
                self._warn_idle('camera', 'video', self._port)
                continue
            except zmq.ZMQError as e:
                log.error(f'ZMQ recv error: {e}')
                break

            self.get_logger().debug(f'Received {len(raw)} bytes from ZMQ')
            proto = camera_frame_pb2.CameraFrame()
            proto.ParseFromString(raw)

            self._warn_skew('camera', proto.timestamp_ns)

            ros_msg = CompressedImage()
            ros_msg.header.stamp = ns_to_ros_time(proto.timestamp_ns)
            ros_msg.header.frame_id = 'pi_camera'
            ros_msg.format = 'jpeg'
            ros_msg.data = list(proto.jpeg_data)

            self.camera_pub.publish(ros_msg)

    def _audio_recv_loop(self):
        sock = self._zmq_context.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT_MS)
        sock.connect(f'tcp://{self._pi_host}:{self._audio_port}')

        last_ts_ns = 0
        chunks = 0
        gap_warnings = 0
        last_stats_t = self.get_clock().now().nanoseconds

        while rclpy.ok():
            try:
                raw = sock.recv()
            except zmq.Again:
                self._warn_idle('audio', 'audio', self._audio_port)
                continue
            except zmq.ZMQError as e:
                log.error(f'ZMQ audio recv error: {e}')
                break

            proto = audio_chunk_pb2.AudioChunk()
            proto.ParseFromString(raw)

            bytes_per_sample = max(1, proto.bit_depth // 8)
            frame_bytes = max(1, proto.channels * bytes_per_sample)
            sample_frames = len(proto.pcm_data) // frame_bytes
            if proto.sample_rate > 0:
                expected_delta_ns = int(sample_frames * 1_000_000_000 / proto.sample_rate)
            else:
                expected_delta_ns = 0

            if last_ts_ns and expected_delta_ns > 0:
                delta_ns = proto.timestamp_ns - last_ts_ns
                if delta_ns > int(expected_delta_ns * 1.5):
                    gap_warnings += 1
                    self.get_logger().warning(
                        f'Audio timestamp gap: delta={delta_ns / 1e6:.2f}ms expected={expected_delta_ns / 1e6:.2f}ms'
                    )
            last_ts_ns = proto.timestamp_ns
            chunks += 1

            self._warn_skew('audio', proto.timestamp_ns)

            ros_msg = RawAudio()
            ros_msg.timestamp = ns_to_ros_time(proto.timestamp_ns)
            ros_msg.data = proto.pcm_data
            ros_msg.format = 'pcm-s16'
            ros_msg.sample_rate = proto.sample_rate
            ros_msg.number_of_channels = proto.channels
            self.audio_pub.publish(ros_msg)

            now_ns = self.get_clock().now().nanoseconds
            if now_ns - last_stats_t >= 1_000_000_000:
                self.get_logger().info(f'Audio rx stats: chunks={chunks}/s gaps={gap_warnings}')
                chunks = 0
                gap_warnings = 0
                last_stats_t = now_ns

    def _log_throttled(self, key: str, message: str, *, error: bool = False) -> None:
        """Emit ``message`` at most once per :data:`WARN_INTERVAL_NS` per ``key``."""
        if not self._throttle.should_emit(key, self.get_clock().now().nanoseconds):
            return
        logger = self.get_logger()
        (logger.error if error else logger.warning)(message)

    def _warn_skew(self, stream: str, stamp_ns: int) -> None:
        """
        Warn that a Pi stamp is too far from this host's clock.

        Warns only — the stamp is published unchanged. Silently correcting it would hide the two
        failures this catches: a Pi whose clock has drifted off the ROS host, and a stamp still
        carrying a device time base (a boottime counter reads as ~1970 here). Both make every
        downstream latency figure meaningless, so they have to be loud.
        """
        now_ns = self.get_clock().now().nanoseconds
        if not stamp_is_skewed(now_ns, stamp_ns, self._max_clock_skew_s):
            return
        self._log_throttled(
            f'skew:{stream}',
            f"Pi {stream} stamps are {(now_ns - stamp_ns) / 1e9:+.3f}s off this host's clock "
            f'(limit {self._max_clock_skew_s:.3f}s). The stream is being published unchanged, '
            f'but every latency derived from it is wrong. Check the Pi is chrony-synced to '
            f'this host — see docs/pi-provisioning.md, "Clock sync".',
            error=True,
        )

    def _warn_idle(self, stream: str, kind: str, port: int) -> None:
        """Warn that no Pi frames are arriving."""
        self._log_throttled(
            f'idle:{stream}',
            f'No {kind} frames from the Pi at tcp://{self._pi_host}:{port} '
            f'— is the Pi stream running (`polyumi-pi stream`)?',
        )

    def destroy_node(self):
        """Terminate ZMQ resources before shutting down the ROS2 node."""
        self._zmq_context.term()
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Receive frames from pi_streamer and publish to ROS2."""
    rclpy.init()
    node = PiReceiverNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

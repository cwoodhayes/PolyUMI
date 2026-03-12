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
import zmq
from builtin_interfaces.msg import Time
from polyumi_pi_msgs import audio_chunk_pb2, camera_frame_pb2
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('pi_receiver_node')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ns_to_ros_time(t_ns: int) -> Time:
    """Convert a Unix timestamp in nanoseconds to a ROS2 Time message."""
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

        self._pi_host = (
            self.get_parameter('pi_host').get_parameter_value().string_value
        )
        self._port = (
            self.get_parameter('port').get_parameter_value().integer_value
        )
        self._audio_port = (
            self.get_parameter('audio_port')
            .get_parameter_value()
            .integer_value
        )

        self.camera_pub = self.create_publisher(
            CompressedImage,
            'camera/image/compressed',
            qos_profile=10,
        )

        self._zmq_context = zmq.Context()

        recv_thread = threading.Thread(
            target=self._camera_recv_loop, daemon=True
        )
        recv_thread.start()

        audio_recv_thread = threading.Thread(
            target=self._audio_recv_loop, daemon=True
        )
        audio_recv_thread.start()

        self.get_logger().info(
            f'Receiving from tcp://{self._pi_host}:{self._port}, '
            f'publishing on /pi/camera/image/compressed'
        )
        self.get_logger().info(
            f'Receiving audio from tcp://{self._pi_host}:{self._audio_port}'
        )

    def _camera_recv_loop(self):
        sock = self._zmq_context.socket(zmq.PULL)
        sock.connect(f'tcp://{self._pi_host}:{self._port}')

        while rclpy.ok():
            try:
                raw = sock.recv()
            except zmq.ZMQError as e:
                log.error(f'ZMQ recv error: {e}')
                break

            self.get_logger().debug(f'Received {len(raw)} bytes from ZMQ')
            proto = camera_frame_pb2.CameraFrame()
            proto.ParseFromString(raw)

            ros_msg = CompressedImage()
            ros_msg.header.stamp = ns_to_ros_time(proto.timestamp_ns)
            ros_msg.header.frame_id = 'pi_camera'
            ros_msg.format = 'jpeg'
            ros_msg.data = list(proto.jpeg_data)

            self.camera_pub.publish(ros_msg)

    def _audio_recv_loop(self):
        sock = self._zmq_context.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b'')
        sock.connect(f'tcp://{self._pi_host}:{self._audio_port}')

        while rclpy.ok():
            try:
                raw = sock.recv()
            except zmq.ZMQError as e:
                log.error(f'ZMQ audio recv error: {e}')
                break

            proto = audio_chunk_pb2.AudioChunk()
            proto.ParseFromString(raw)
            self.get_logger().info(
                'AudioChunk received: '
                f'ts={proto.timestamp_ns} '
                f'bytes={len(proto.pcm_data)} '
                f'sr={proto.sample_rate} '
                f'ch={proto.channels}'
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

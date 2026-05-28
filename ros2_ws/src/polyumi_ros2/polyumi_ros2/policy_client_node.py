r"""
ROS2 node that drives the Franka arm using a remote diffusion policy inference server.

At each control tick the node:
  1. Reads the latest wrist camera image and end-effector pose.
  2. Maintains a short history window (n_obs_steps).
  3. POSTs observations to /predict_cartesian/ on the inference server.
  4. Logs the returned action (Phase 1). Motion execution is added in Phase 2.

Usage:
    ros2 run polyumi_ros2 policy_client_node
    ros2 run polyumi_ros2 policy_client_node --ros-args \\
        -p inference_server_url:=http://192.168.1.10:8000/predict_cartesian/
"""

import json
import threading

from collections import deque

import cv2
import numpy as np
import rclpy
import rclpy.time
import urllib.request
import urllib.error
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException  # type: ignore[attr-defined]


class PolicyClientNode(Node):
    """Buffer observations and call the remote inference server at a fixed rate."""

    def __init__(self):
        """Declare parameters, create subscribers, TF buffer, and control timer."""
        super().__init__('policy_client_node')

        self.declare_parameter('inference_server_url', 'http://localhost:8000/predict_cartesian/')
        self.declare_parameter('n_obs_steps', 2)
        self.declare_parameter('image_topic', '/gopro/image_raw')
        self.declare_parameter('control_hz', 10.0)
        self.declare_parameter('image_width', 256)
        self.declare_parameter('image_height', 256)

        self._url = self.get_parameter('inference_server_url').get_parameter_value().string_value
        self._n_obs_steps = self.get_parameter('n_obs_steps').get_parameter_value().integer_value
        self._image_w = self.get_parameter('image_width').get_parameter_value().integer_value
        self._image_h = self.get_parameter('image_height').get_parameter_value().integer_value
        control_hz = self.get_parameter('control_hz').get_parameter_value().double_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        # History buffers — each entry: (image_float32 [H,W,C], agent_pos [8])
        self._obs_buffer: deque = deque(maxlen=self._n_obs_steps)
        self._latest_image: np.ndarray | None = None
        self._latest_image_lock = threading.Lock()

        # TF
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Subscribers
        self.create_subscription(Image, image_topic, self._image_cb, 10)

        # Control timer
        period = 1.0 / control_hz
        self.create_timer(period, self._control_tick)

        # Throttle for "buffer not full" warning
        self._last_warn_t: rclpy.time.Time | None = None

        self.get_logger().info(f'policy_client_node started — server: {self._url}')

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image) -> None:
        """Convert incoming ROS image to float32 numpy array and cache it."""
        # Support rgb8 and bgr8 encodings
        dtype = np.uint8
        channels = 3
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1].copy()  # BGR → RGB
        resized = cv2.resize(img, (self._image_w, self._image_h), interpolation=cv2.INTER_LINEAR)
        float_img = resized.astype(np.float32) / 255.0
        with self._latest_image_lock:
            self._latest_image = float_img

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_tick(self) -> None:
        """Assemble one observation, fill buffer, POST to inference server."""
        # --- 1. Get latest image ---
        with self._latest_image_lock:
            image = self._latest_image
        if image is None:
            self._warn_throttled('Waiting for first camera image')
            return

        # --- 2. Get EEF pose from TF ---
        agent_pos = self._lookup_agent_pos()
        if agent_pos is None:
            return  # warning already logged inside

        # --- 3. Append to history buffer ---
        self._obs_buffer.append((image, agent_pos))
        if len(self._obs_buffer) < self._n_obs_steps:
            self._warn_throttled(
                f'Observation buffer filling ({len(self._obs_buffer)}/{self._n_obs_steps})'
            )
            return

        # --- 4. Serialize and POST ---
        images = [obs[0].tolist() for obs in self._obs_buffer]
        poses = [obs[1].tolist() for obs in self._obs_buffer]
        payload = {
            'n_obs_steps': self._n_obs_steps,
            'n_action_steps': 1,
            'observations': {
                'image': images,
                'agent_pos': poses,
            },
        }
        self._post_and_log(payload)

    def _lookup_agent_pos(self) -> np.ndarray | None:
        """Look up panda_EE in panda_link0 and return [x,y,z,qx,qy,qz,qw, gripper=0]."""
        try:
            tf = self._tf_buffer.lookup_transform(
                'panda_link0', 'panda_EE', self.get_clock().now()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self._warn_throttled(f'TF lookup failed: {e}')
            return None

        t = tf.transform.translation
        r = tf.transform.rotation
        # gripper_width placeholder — replaced in Phase 2 with real joint state subscriber
        gripper_width = 0.0
        return np.array([t.x, t.y, t.z, r.x, r.y, r.z, r.w, gripper_width], dtype=np.float64)

    def _post_and_log(self, payload: dict) -> None:
        """POST payload to inference server and log the returned action."""
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                result = json.loads(resp.read())
            action = result['actions'][0]
            self.get_logger().info(
                f'action x={action[0]:.4f} y={action[1]:.4f} z={action[2]:.4f} '
                f'grip={action[7]:.3f}'
            )
        except urllib.error.URLError as e:
            self.get_logger().error(f'Inference server unreachable: {e}')
        except Exception as e:
            self.get_logger().error(f'POST failed: {e}')

    def _warn_throttled(self, msg: str) -> None:
        """Log a warning at most once per second."""
        now = self.get_clock().now()
        if self._last_warn_t is None or (now - self._last_warn_t).nanoseconds >= 1_000_000_000:
            self.get_logger().warn(msg)
            self._last_warn_t = now


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Start the policy client node."""
    rclpy.init()
    node = PolicyClientNode()
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

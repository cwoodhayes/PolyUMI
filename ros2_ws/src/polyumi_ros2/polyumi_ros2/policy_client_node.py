r"""
ROS2 node that drives the Franka arm using a remote diffusion policy inference server.

At each control tick the node:
  1. Reads the latest wrist camera image and end-effector pose.
  2. Maintains a short history window (n_obs_steps).
  3. POSTs observations to /predict_cartesian/ on the inference server, requesting an
     n_action_steps-length action chunk.
  4. Logs the chunk. If execute_motion is set, publishes the whole chunk as a PoseArray
     on /polyumi/target_poses for the NUC-side fr3_moveit_bridge to plan+execute as one
     Cartesian path (receding-horizon control) — see docs/crb-fr3-inference.md.

Usage:
    ros2 run polyumi_ros2 policy_client_node
    ros2 run polyumi_ros2 policy_client_node --ros-args \\
        -p inference_server_url:=http://192.168.1.10:8000/predict_cartesian/
"""

import base64
import json
import threading

from collections import deque

import cv2
import numpy as np
import rclpy
import rclpy.time
import urllib.request
import urllib.error
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException  # type: ignore[attr-defined]

from geometry_msgs.msg import Pose, PoseArray


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
        # Frame IDs for the EEF pose lookup. Defaults match the FR3 TF tree; on a
        # different arm override base_frame / eef_frame instead of editing code.
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('eef_frame', 'fr3_hand_tcp')
        # Motion execution (Phase 2). Off by default for safety: the node logs actions
        # but does NOT publish target poses unless execute_motion is explicitly enabled.
        # Planning params (group, velocity scaling) live on the NUC bridge, not here.
        self.declare_parameter('execute_motion', False)
        # Action-chunk size requested from the server and published for execution as one
        # multi-waypoint Cartesian path. UMI/DP-style receding-horizon control: 1 would mean
        # a discrete hop every control tick, which the arm can't track in real time — the
        # bridge's skip-while-busy would drop almost every tick. A full chunk lets move_group
        # plan one smooth path instead. Must be <= the model's n_action_steps (dummy: 8).
        self.declare_parameter('n_action_steps', 8)
        # Per-component system latencies (seconds), as measured by calibration scripts.
        # Loaded from config/latency.yaml via the inference launch file; not yet consumed
        # for compensation, just plumbed through and logged.
        self.declare_parameter('latency.gopro_s', 0.0)
        self.declare_parameter('latency.finger_cam', 0.0)
        self.declare_parameter('latency.piezo_mic', 0.0)
        self.declare_parameter('latency.proprio', 0.0)
        self.declare_parameter('latency.arm_exec', 0.0)

        self._url = self.get_parameter('inference_server_url').get_parameter_value().string_value
        self._n_obs_steps = self.get_parameter('n_obs_steps').get_parameter_value().integer_value
        self._image_w = self.get_parameter('image_width').get_parameter_value().integer_value
        self._image_h = self.get_parameter('image_height').get_parameter_value().integer_value
        self._base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self._eef_frame = self.get_parameter('eef_frame').get_parameter_value().string_value
        self._execute_motion = self.get_parameter('execute_motion').get_parameter_value().bool_value
        self._n_action_steps = self.get_parameter('n_action_steps').get_parameter_value().integer_value
        control_hz = self.get_parameter('control_hz').get_parameter_value().double_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self._latency = {
            'gopro_s': self.get_parameter('latency.gopro_s').get_parameter_value().double_value,
            'finger_cam': self.get_parameter('latency.finger_cam').get_parameter_value().double_value,
            'piezo_mic': self.get_parameter('latency.piezo_mic').get_parameter_value().double_value,
            'proprio': self.get_parameter('latency.proprio').get_parameter_value().double_value,
            'arm_exec': self.get_parameter('latency.arm_exec').get_parameter_value().double_value,
        }

        # History buffers — each entry: (image_float32 [H,W,C], agent_pos [8])
        self._obs_buffer: deque = deque(maxlen=self._n_obs_steps)
        self._latest_image: np.ndarray | None = None
        self._latest_image_lock = threading.Lock()

        # TF
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Motion execution (Phase 2). The MoveIt calls run in a bridge node ON THE NUC
        # (fr3_moveit_bridge), not here: the laptop (rmw_cyclonedds 4.x, Kilted) and NUC
        # (rmw 1.x, Humble) can exchange small messages but corrupt large MoveIt action
        # goals across the rmw-major boundary. So when execution is enabled we just publish
        # the target EEF pose chunk (PoseArray); the NUC bridge subscribes and plans+executes
        # the whole chunk as one Cartesian path via its local move_group.
        self._target_pub = None
        if self._execute_motion:
            self._target_pub = self.create_publisher(PoseArray, '/polyumi/target_poses', 10)

        # Subscribers
        self.create_subscription(Image, image_topic, self._image_cb, 10)

        # Control timer — exclusive callback group ensures only one tick (and its
        # blocking POST) runs at a time; an in-flight tick causes the next one to
        # be skipped rather than overlapping.
        self._tick_lock = threading.Lock()
        period = 1.0 / control_hz
        self.create_timer(period, self._control_tick, callback_group=MutuallyExclusiveCallbackGroup())

        # Throttle for "buffer not full" warning
        self._last_warn_t: rclpy.time.Time | None = None

        mode = 'EXECUTE (arm will move)' if self._execute_motion else 'log-only (no motion)'
        self.get_logger().info(f'policy_client_node started — server: {self._url} — mode: {mode}')
        latency_str = ' '.join(f'{name}={seconds}s' for name, seconds in self._latency.items())
        self.get_logger().info(f'latency config — {latency_str}')

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image) -> None:
        """Convert incoming ROS image to float32 numpy array and cache it."""
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError(f'Unsupported image encoding {msg.encoding!r}; expected rgb8 or bgr8')
        if msg.step != msg.width * 3:
            raise ValueError(f'Unsupported row stride: step={msg.step} != width*3={msg.width * 3}')
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
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
        """
        Assemble one observation, fill buffer, POST to inference server.

        If the previous tick's POST is still in flight, this tick is skipped
        (and a warning logged) rather than overlapping with it.
        """
        if not self._tick_lock.acquire(blocking=False):
            self.get_logger().warn('Dropped control tick: previous POST to inference server still in flight')
            return
        try:
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
                self._warn_throttled(f'Observation buffer filling ({len(self._obs_buffer)}/{self._n_obs_steps})')
                return

            # --- 4. Serialize and POST ---
            # Images go as base64-encoded raw bytes (+ dtype/shape) rather than nested-list
            # JSON, which is ~1.5MB+ per frame and slow to encode/decode at 10 Hz.
            # agent_pos is tiny (n_obs_steps x 8 floats) so it stays a plain list.
            image_stack = np.stack([obs[0] for obs in self._obs_buffer])
            poses = [obs[1].tolist() for obs in self._obs_buffer]
            payload = {
                'n_obs_steps': self._n_obs_steps,
                'n_action_steps': self._n_action_steps,
                'observations': {
                    'image': {
                        'dtype': str(image_stack.dtype),
                        'shape': list(image_stack.shape),
                        'data': base64.b64encode(image_stack.tobytes()).decode('ascii'),
                    },
                    'agent_pos': poses,
                },
            }
            self._post_and_act(payload)
        finally:
            self._tick_lock.release()

    def _lookup_agent_pos(self) -> np.ndarray | None:
        """Look up eef_frame in base_frame and return [x,y,z,qx,qy,qz,qw, gripper=0]."""
        try:
            # rclpy.time.Time() (zero) requests the latest available transform, avoiding
            # ExtrapolationException when the buffer hasn't caught up to get_clock().now().
            tf = self._tf_buffer.lookup_transform(self._base_frame, self._eef_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self._warn_throttled(f'TF lookup failed: {e}')
            return None

        t = tf.transform.translation
        r = tf.transform.rotation
        # gripper_width placeholder — replaced in Phase 2 with real joint state subscriber
        gripper_width = 0.0
        return np.array([t.x, t.y, t.z, r.x, r.y, r.z, r.w, gripper_width], dtype=np.float64)

    def _post_and_act(self, payload: dict) -> None:
        """POST payload to the inference server, log the returned action, and optionally execute it."""
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
            actions = result['actions']
            first = actions[0]
            self.get_logger().info(
                f'action chunk n={len(actions)} first: x={first[0]:.4f} y={first[1]:.4f} '
                f'z={first[2]:.4f} grip={first[7]:.3f}'
            )
        except urllib.error.URLError as e:
            self.get_logger().error(f'Inference server unreachable: {e}')
            return
        except Exception as e:
            self.get_logger().error(f'POST failed: {e}')
            return

        # Phase 2: publish the whole action chunk for the NUC bridge to plan+execute as one
        # Cartesian path (receding-horizon control). Non-blocking (unlike a direct MoveIt
        # call): the NUC bridge does its own skip-while-busy, so at worst it drops chunks
        # that arrive mid-motion. Gripper (action[7]) is deferred; only xyz+quat is published.
        if self._target_pub is not None:
            self._target_pub.publish(self._actions_to_pose_array(actions))

    def _actions_to_pose_array(self, actions) -> PoseArray:
        """Build a PoseArray in base_frame from a list of 8-vector actions [x,y,z,qx,qy,qz,qw,grip]."""
        poses = []
        for action in actions:
            pose = Pose()
            pose.position.x = float(action[0])
            pose.position.y = float(action[1])
            pose.position.z = float(action[2])
            pose.orientation.x = float(action[3])
            pose.orientation.y = float(action[4])
            pose.orientation.z = float(action[5])
            pose.orientation.w = float(action[6])
            poses.append(pose)

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.poses = poses
        return msg

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

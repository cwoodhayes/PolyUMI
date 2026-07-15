r"""
ROS2 node that drives the Franka arm using a remote diffusion policy inference server.

At each control tick the node:
  1. Reads the latest wrist camera image and a latency-compensated end-effector pose (looked
     up in TF at the frame's own stamp - latency.gopro, to align with when that image was
     actually captured).
  2. Maintains a short history window (n_obs_steps).
  3. POSTs observations to /predict_cartesian/ on the inference server, requesting an
     n_action_steps-length action chunk.
  4. Drops the leading actions of the returned chunk that are already stale by the time the
     arm could act on them (observation + inference + arm-execution latency).
  5. Logs the chunk. If execute_motion is set, publishes the remaining chunk as a PoseArray
     on /polyumi/target_poses for the NUC-side fr3_moveit_bridge to plan+execute as one
     Cartesian path (receding-horizon control) — see docs/crb-fr3-inference.md.

Usage:
    ros2 run polyumi_ros2 policy_client_node
    ros2 run polyumi_ros2 policy_client_node --ros-args \\
        -p inference_server_url:=http://192.168.1.10:8000/predict_cartesian/
"""

import base64
import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections import deque

import cv2
import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException  # type: ignore[attr-defined]


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
        # Per-component system latencies (seconds), as measured by calibration scripts and
        # loaded from config/latency.yaml via the inference launch file. Only gopro is
        # currently consumed (see _lookup_agent_pos); the rest are logged for now.
        self.declare_parameter('latency.gopro', 0.0)
        self.declare_parameter('latency.finger_cam', 0.0)
        self.declare_parameter('latency.piezo_mic', 0.0)
        self.declare_parameter('latency.proprio', 0.0)
        self.declare_parameter('latency.arm_exec', 0.0)
        # How far back (seconds) the EE-pose TF buffer retains history — must be >= the
        # largest latency being compensated for (see _lookup_agent_pos).
        self.declare_parameter('buffers.ee_pose_s', 1.0)

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
            'gopro': self.get_parameter('latency.gopro').get_parameter_value().double_value,
            'finger_cam': self.get_parameter('latency.finger_cam').get_parameter_value().double_value,
            'piezo_mic': self.get_parameter('latency.piezo_mic').get_parameter_value().double_value,
            'proprio': self.get_parameter('latency.proprio').get_parameter_value().double_value,
            'arm_exec': self.get_parameter('latency.arm_exec').get_parameter_value().double_value,
        }
        self._ee_pose_buffer_s = self.get_parameter('buffers.ee_pose_s').get_parameter_value().double_value

        # Observation age is no longer summed from constants — it's measured from the frame's
        # own stamp (see _n_stale_actions). latency.gopro still converts that stamp to a true
        # capture instant, and is the only delayed modality the policy consumes; once
        # finger_cam/piezo_mic feed it too, the capture instant becomes the oldest across them,
        # since an observation is only as fresh as its slowest stream.
        # latency_act — delay between publishing a target and the arm actually moving.
        self._latency_act = self._latency['arm_exec']
        # Spacing between consecutive actions within a chunk. Assumes the policy's action
        # horizon runs at the observation/control rate (standard for UMI/diffusion policy);
        # if a model is ever trained at a different action rate this needs its own parameter.
        self._action_dt = 1.0 / control_hz

        # History buffers — each entry: (image_float32 [H,W,C], agent_pos [8])
        self._obs_buffer: deque = deque(maxlen=self._n_obs_steps)
        self._latest_image: np.ndarray | None = None
        self._latest_image_stamp: rclpy.time.Time | None = None
        self._latest_image_lock = threading.Lock()
        # Reject a cached frame older than this at tick time. Sized to two camera periods at
        # the 60 Hz v4l2 rate, floored so a slow tick doesn't trip it; a frame older than this
        # means the capture pipeline stalled, and pairing it with a fresh pose would silently
        # feed the policy a mismatched observation.
        self._max_image_age_s = max(2.0 / 60.0, 0.5 / control_hz)

        # TF — cache_time sized from buffers.ee_pose_s so a pose from up to that far back can
        # still be looked up (needed to time-align with the delayed gopro frame; see
        # _lookup_agent_pos). tf2's buffer already interpolates (linear + slerp) between the
        # two nearest cached transforms for a historical lookup_transform() call, so there's
        # no need for a separate hand-rolled pose buffer.
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=self._ee_pose_buffer_s))
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
        self.get_logger().info(f'latency config — {latency_str} (ee_pose buffer: {self._ee_pose_buffer_s}s)')
        self.get_logger().info(
            f'latency budget — measured observation age (capture→response) + '
            f'act={self._latency_act}s vs action_dt={self._action_dt}s'
        )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image) -> None:
        """Convert incoming ROS image to float32 numpy array and cache it with its stamp."""
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
            # Keep the frame's own stamp: the pose lookup must align to when THIS frame was
            # captured, not to when the control tick happens to run. The camera publishes at
            # 60 Hz while the tick runs at control_hz, so a cached frame is already up to one
            # camera period old before the tick even fires — and if the v4l2 pipeline stalls,
            # unboundedly older, with no way to notice. See _lookup_agent_pos.
            self._latest_image_stamp = rclpy.time.Time.from_msg(msg.header.stamp)

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
                image_stamp = self._latest_image_stamp
            if image is None or image_stamp is None:
                self._warn_throttled('Waiting for first camera image')
                return

            # Guard against pairing a stale frame with a fresh pose (see _max_image_age_s).
            image_age_s = (self.get_clock().now() - image_stamp).nanoseconds * 1e-9
            if image_age_s > self._max_image_age_s:
                self._warn_throttled(
                    f'Dropped control tick: newest camera frame is {image_age_s * 1e3:.0f} ms old '
                    f'(limit {self._max_image_age_s * 1e3:.0f} ms) — capture pipeline stalled?'
                )
                return

            # --- 2. Get EEF pose from TF, aligned to this frame's capture instant ---
            agent_pos = self._lookup_agent_pos(image_stamp)
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
            # t_obs: when this frame was actually captured, i.e. the instant action[0] targets.
            self._post_and_act(payload, image_stamp - Duration(seconds=self._latency['gopro']))
        finally:
            self._tick_lock.release()

    def _lookup_agent_pos(self, image_stamp: rclpy.time.Time) -> np.ndarray | None:
        """
        Look up eef_frame in base_frame, time-aligned to the gopro frame, and return the pose.

        The gopro image being paired with this pose was captured ~latency.gopro seconds before
        ``image_stamp`` (the v4l2 driver stamps a frame when it dequeues the buffer, which is
        already downstream of GoPro encode + HDMI out + capture card + USB transfer), so we
        look up the EE pose as of that same past instant — not the current one — to keep image
        and proprio in sync. This mirrors UMI's scheme, where each sensor's receive timestamp
        is corrected back to true capture time and the low-dim streams are then interpolated
        onto the camera's corrected clock.

        Anchoring on the frame's own stamp rather than the tick's wall clock matters because
        the two are decoupled: the camera runs at 60 Hz and the tick at control_hz, so
        `now()` overstates the frame's freshness by a jittering 0–1 camera periods.

        tf2's Buffer interpolates (linear + slerp) between the two nearest cached transforms
        automatically; buffers.ee_pose_s sizes the buffer's cache_time so the lookup stays in
        range.

        :param image_stamp: header stamp of the camera frame this pose will be paired with.
        """
        target_time = image_stamp - Duration(seconds=self._latency['gopro']) + \
            Duration(seconds=self._latency['proprio'])
        try:
            tf = self._tf_buffer.lookup_transform(self._base_frame, self._eef_frame, target_time)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self._warn_throttled(f'TF lookup failed: {e}')
            return None

        t = tf.transform.translation
        r = tf.transform.rotation
        # gripper_width placeholder — replaced in Phase 2 with real joint state subscriber
        gripper_width = 0.0
        return np.array([t.x, t.y, t.z, r.x, r.y, r.z, r.w, gripper_width], dtype=np.float64)

    def _n_stale_actions(self, t_obs: rclpy.time.Time) -> int:
        """
        Count the leading actions in a chunk that are already in the past by execution time.

        Action i in a chunk is the policy's target for t_obs + i * action_dt, where t_obs is
        the instant the observation was captured. Everything between that instant and the arm
        actually moving makes the leading actions stale: the frame's transit through the
        capture pipeline, the tick's own serialization work, the server's round trip, and
        latency_act before the arm starts moving. Actions whose target instant falls inside
        that window have already elapsed — executing them would drag the arm backwards through
        the trajectory — so skip to the first one still in the future.

        Called after the server responds, so ``now() - t_obs`` measures every delay up to this
        point directly (capture pipeline + tick + inference round trip) instead of summing
        assumed constants for them; only latency_act is still in the future and must be added.

        :param t_obs: instant the observation was captured (image stamp - latency.gopro).
        :return: number of actions to drop from the front of the chunk.
        """
        elapsed_since_obs = (self.get_clock().now() - t_obs).nanoseconds * 1e-9
        total_latency = elapsed_since_obs + self._latency_act
        return math.ceil(total_latency / self._action_dt)

    def _post_and_act(self, payload: dict, t_obs: rclpy.time.Time) -> None:
        """
        POST payload to the inference server, log the returned action, and optionally execute it.

        :param payload: request body for /predict_cartesian/.
        :param t_obs: instant the observation was captured, used to drop stale actions.
        """
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            t_sent = time.monotonic()
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                result = json.loads(resp.read())
            latency_inference = time.monotonic() - t_sent
            actions = result['actions']
        except urllib.error.URLError as e:
            self.get_logger().error(f'Inference server unreachable: {e}')
            return
        except Exception as e:
            self.get_logger().error(f'POST failed: {e}')
            return

        # Drop the leading actions that refer to instants already elapsed by the time the arm
        # can act on them, so execution starts from the first still-future waypoint.
        n_stale = self._n_stale_actions(t_obs)
        n_received = len(actions)
        actions = actions[n_stale:]
        if not actions:
            age_s = (self.get_clock().now() - t_obs).nanoseconds * 1e-9
            self._warn_throttled(
                f'Whole action chunk stale: dropped all {n_received} actions '
                f'(observation is {age_s:.3f}s old, of which inference={latency_inference:.3f}s; '
                f'plus act={self._latency_act:.3f}s exceeds chunk span '
                f'{n_received * self._action_dt:.3f}s). Raise n_action_steps or reduce latency.'
            )
            return

        first = actions[0]
        self.get_logger().info(
            f'action chunk n={len(actions)} (dropped {n_stale}/{n_received} stale, '
            f'inference={latency_inference * 1000:.0f}ms) first: x={first[0]:.4f} y={first[1]:.4f} '
            f'z={first[2]:.4f} grip={first[7]:.3f}'
        )

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

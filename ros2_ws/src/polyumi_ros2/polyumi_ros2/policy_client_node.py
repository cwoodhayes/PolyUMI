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

    def __init__(self, **kwargs):
        """
        Declare parameters, create subscribers, TF buffer, and control timer.

        :param kwargs: forwarded to rclpy's Node — notably ``parameter_overrides``, which lets
            tests construct the node with specific values without going through a launch file.
        """
        super().__init__('policy_client_node', **kwargs)

        self.declare_parameter('inference_server_url', 'http://localhost:8000/predict_cartesian/')
        self.declare_parameter('n_obs_steps', 2)
        self.declare_parameter('image_topic', '/gopro/image_raw')
        self.declare_parameter('control_hz', 10.0)
        # 224 matches the model's shape_meta (camera0_rgb [3,224,224]); the DP exporter squashes
        # frames to 224 too, so the client's resize reproduces training's aspect handling.
        self.declare_parameter('image_width', 224)
        self.declare_parameter('image_height', 224)
        # Max age (s) of the newest cached camera frame before a tick is dropped as a stalled
        # capture pipeline. 0.0 = auto: max(2 camera periods @ 60 Hz, half a control period). The
        # auto value assumes a ~60 Hz camera; a slower path (e.g. an Elgato capture card doing a
        # 1080p software YUYV→RGB conversion, which adds ~200 ms of stamp-to-usable latency) needs
        # a larger value. Raising it only tolerates older frames — image and pose stay aligned to
        # the frame's own capture stamp regardless (see _lookup_agent_pos).
        self.declare_parameter('max_image_age_s', 0.0)
        # Frame IDs for the EEF pose lookup. Defaults match the FR3 TF tree; on a
        # different arm override base_frame / eef_frame instead of editing code.
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('eef_frame', 'fr3_hand_tcp')
        # Motion execution (Phase 2). Off by default for safety: the node logs actions
        # but does NOT publish target poses unless execute_motion is explicitly enabled.
        # Planning params (group, velocity scaling) live on the NUC bridge, not here.
        self.declare_parameter('execute_motion', False)
        # Viz-only preview: publish every commanded chunk as a PoseArray on
        # /polyumi/target_poses_preview regardless of execute_motion, so the motion can be seen in
        # Foxglove/RViz without the arm moving (the NUC bridge subscribes only to the execution
        # topic /polyumi/target_poses, never this one). On by default — it moves nothing.
        self.declare_parameter('publish_preview', True)
        # HTTP timeout (s) for the inference POST. Real diffusion inference over LAN is slower
        # than the trivial dummy server; 0.5 s was fine for the dummy but risks timing out every
        # tick against the real model. An overrun just skips the tick (see the tick lock).
        self.declare_parameter('post_timeout_s', 1.0)
        # Action-chunk size requested from the server and published for execution as one
        # multi-waypoint Cartesian path. UMI/DP-style receding-horizon control: 1 would mean
        # a discrete hop every control tick, which the arm can't track in real time — the
        # bridge's skip-while-busy would drop almost every tick. A full chunk lets move_group
        # plan one smooth path instead. Must be <= the model's n_action_steps (dummy: 8).
        self.declare_parameter('n_action_steps', 8)
        # Per-component system latencies (seconds), loaded from config/latency.yaml via the
        # inference launch file. NONE of them have been measured yet — see that file and
        # blocking issue 2 in docs/franka-inference-bringup.md. gopro and proprio are consumed
        # by _lookup_agent_pos, arm_exec by _n_stale_actions; finger_cam and piezo_mic are
        # declared but unused until the policy takes tactile input.
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
        max_image_age_s = self.get_parameter('max_image_age_s').get_parameter_value().double_value
        self._base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self._eef_frame = self.get_parameter('eef_frame').get_parameter_value().string_value
        self._execute_motion = self.get_parameter('execute_motion').get_parameter_value().bool_value
        self._publish_preview = self.get_parameter('publish_preview').get_parameter_value().bool_value
        self._post_timeout_s = self.get_parameter('post_timeout_s').get_parameter_value().double_value
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
        self._validate_params(control_hz)

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
        # Reject a cached frame older than this at tick time; a frame older than this means the
        # capture pipeline stalled. The auto default (max_image_age_s <= 0) is two camera periods
        # at the 60 Hz v4l2 rate, floored at half a control period so a slow tick doesn't trip it;
        # override the param for slower camera paths (see the param declaration).
        self._max_image_age_s = (
            max_image_age_s if max_image_age_s > 0 else max(2.0 / 60.0, 0.5 / control_hz)
        )

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

        # Viz-only preview publisher (always on when publish_preview). Shows every commanded chunk
        # in Foxglove/RViz without moving the arm: the NUC bridge subscribes only to the execution
        # topic /polyumi/target_poses, never this one.
        self._preview_pub = None
        if self._publish_preview:
            self._preview_pub = self.create_publisher(PoseArray, '/polyumi/target_poses_preview', 10)

        # Episode-start /reset. The server needs the episode-start EEF pose for
        # robot0_eef_rot_axis_angle_wrt_start; sent once on the first full-buffer tick. The reset
        # URL is derived from the predict URL's base so one param configures both endpoints.
        self._reset_url = self._url.split('/predict_cartesian')[0] + '/reset'
        self._episode_reset_done = False

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
        preview = 'on (/polyumi/target_poses_preview)' if self._publish_preview else 'off'
        self.get_logger().info(
            f'policy_client_node started — server: {self._url} — mode: {mode} — preview: {preview}'
        )
        latency_str = ' '.join(f'{name}={seconds}s' for name, seconds in self._latency.items())
        self.get_logger().info(
            f'latency config — {latency_str} (ee_pose buffer: {self._ee_pose_buffer_s}s, '
            f'max_image_age: {self._max_image_age_s * 1e3:.0f}ms)'
        )
        self.get_logger().info(
            f'latency budget — measured observation age (capture→response) + '
            f'act={self._latency_act}s vs action_dt={self._action_dt}s'
        )

    def _validate_params(self, control_hz: float) -> None:
        """
        Fail fast on parameter values that would corrupt the time math rather than error.

        These all feed divisions and instant arithmetic, where a bad value degrades quietly
        instead of raising: a negative latency puts t_obs in the future and makes the
        stale-action count negative; a non-positive ee_pose buffer leaves TF with no history,
        so every time-aligned lookup fails and the node just logs TF errors forever. A config
        typo should say so at startup, not surface as a mystery three layers down.

        :raises ValueError: on any non-positive rate/buffer or negative latency.
        """
        errors = []
        if control_hz <= 0:
            errors.append(f'control_hz must be > 0, got {control_hz}')
        if self._ee_pose_buffer_s <= 0:
            errors.append(f'buffers.ee_pose_s must be > 0, got {self._ee_pose_buffer_s}')
        if self._n_obs_steps < 1:
            errors.append(f'n_obs_steps must be >= 1, got {self._n_obs_steps}')
        if self._n_action_steps < 1:
            errors.append(f'n_action_steps must be >= 1, got {self._n_action_steps}')
        if self._post_timeout_s <= 0:
            errors.append(f'post_timeout_s must be > 0, got {self._post_timeout_s}')
        for name, seconds in self._latency.items():
            if seconds < 0:
                errors.append(f'latency.{name} must be >= 0, got {seconds}')
        # The TF buffer must reach back at least as far as the instant we look poses up at,
        # or _lookup_agent_pos asks for a transform the buffer has already dropped.
        compensated = self._latency['gopro'] - self._latency['proprio']
        if compensated > self._ee_pose_buffer_s:
            errors.append(
                f'buffers.ee_pose_s ({self._ee_pose_buffer_s}s) must be >= the compensated '
                f'lookup offset (latency.gopro - latency.proprio = {compensated}s), or the '
                f'pose lookup will fall outside the TF buffer'
            )
        if errors:
            raise ValueError('Invalid policy_client_node configuration: ' + '; '.join(errors))

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

            # First full observation marks the episode start: tell the server the start pose once
            # (used for robot0_eef_rot_axis_angle_wrt_start). Retried on failure until it lands.
            if not self._episode_reset_done:
                self._reset_episode(agent_pos)

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

        Clamped at 0: if t_obs somehow lands in the future (clock skew between the camera
        driver's stamps and this node), a negative count would slice from the *end* of the
        chunk — `actions[-3:]` keeps the last three, jumping the arm ahead to the far-future
        tail of the trajectory instead of dropping anything. No upper clamp is needed: a count
        at or past the chunk length yields an empty slice, which the caller already reports.

        :param t_obs: instant the observation was captured (image stamp - latency.gopro).
        :return: number of actions to drop from the front of the chunk, >= 0.
        """
        elapsed_since_obs = (self.get_clock().now() - t_obs).nanoseconds * 1e-9
        total_latency = elapsed_since_obs + self._latency_act
        return max(0, math.ceil(total_latency / self._action_dt))

    def _http_post_json(self, url: str, payload: dict) -> dict | None:
        """POST payload as JSON to url and return the parsed response, or None on failure (logged)."""
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body, headers={'Content-Type': 'application/json'}, method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=self._post_timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            self.get_logger().error(f'POST {url} unreachable: {e}')
        except Exception as e:
            self.get_logger().error(f'POST {url} failed: {e}')
        return None

    def _reset_episode(self, agent_pos: np.ndarray) -> None:
        """Send the episode-start pose to the server's /reset once; retried each tick until it lands."""
        result = self._http_post_json(self._reset_url, {'agent_pos': agent_pos.tolist()})
        if result is not None:
            self._episode_reset_done = True
            self.get_logger().info(f'episode /reset sent (start pose set): {self._reset_url}')
        else:
            self._warn_throttled(
                'episode /reset failed; server will approximate wrt_start with the current pose'
            )

    def _post_and_act(self, payload: dict, t_obs: rclpy.time.Time) -> None:
        """
        POST payload to the inference server, log the returned action, and optionally execute it.

        :param payload: request body for /predict_cartesian/.
        :param t_obs: instant the observation was captured, used to drop stale actions.
        """
        t_sent = time.monotonic()
        result = self._http_post_json(self._url, payload)
        if result is None:
            return
        latency_inference = time.monotonic() - t_sent
        actions = result['actions']

        # Viz-only preview: publish the full commanded chunk (before the stale-drop below) so the
        # motion is visible in Foxglove/RViz even when execute_motion is off or the whole chunk is
        # stale. The NUC bridge never subscribes to this topic, so nothing moves.
        if self._preview_pub is not None:
            self._preview_pub.publish(self._actions_to_pose_array(actions))

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

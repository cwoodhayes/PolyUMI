r"""
ROS2 node that drives the Franka arm using a remote diffusion policy inference server.

At each control tick the node:
  1. Reads the latest wrist camera image and a latency-compensated end-effector pose (looked
     up in TF at the frame's own stamp - latency.gopro, to align with when that image was
     actually captured).
  2. Maintains a short history window (n_obs_steps), appended every tick so the window stays
     dt-spaced regardless of how often inference runs.
  3. Once per ``steps_per_inference`` ticks (a receding-horizon stride — NOT every tick), POSTs
     observations to /predict_cartesian/ on the inference server, requesting an
     n_action_steps-length action chunk. Between inferences the arm keeps executing the
     previously published chunk, matching UMI (infer once per ~steps_per_inference*dt).
  4. Drops the leading actions of the returned chunk that are already stale by the time the
     arm could act on them (observation + inference + arm-execution latency).
  5. Logs the chunk. If execute_motion is set, publishes the remaining chunk on two topics for
     the NUC-side bridges: the pose half as a PoseArray on /polyumi/target_poses (planned and
     executed as one Cartesian path by fr3_moveit_bridge, receding-horizon control), and the
     gripper half as a JointTrajectory on /polyumi/target_gripper (fr3_gripper_bridge). The two
     ride separate channels because a PoseArray cannot carry a width and because the Franka Hand
     is action-only, so it needs a different execution cadence entirely — see
     docs/crb-fr3-inference.md and docs/franka-inference-bringup.md ("Gripper hardware").

Usage:
    ros2 run polyumi_ros2 policy_client_node
    ros2 run polyumi_ros2 policy_client_node --ros-args \\
        -p inference_server_url:=http://192.168.1.10:8000/predict_cartesian/
"""

import base64
import json
import math
import os
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
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException  # type: ignore[attr-defined]
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from polyumi_ros2.camera_preproc import (
    CAMERA0_RGB_INTERPOLATION,
    MAX_BAR_INTENSITY,
    crop_to_source_aspect,
    discarded_bar_intensity,
)
from polyumi_ros2.gripper_map import policy_to_robot_width, robot_to_policy_width
from polyumi_ros2.target_chunk import TargetChunkPublisher, Wire, pose_array

# Name used for the single "joint" in the gripper trajectory chunk. Deliberately NOT a real joint
# name (the FR3's fingers are fr3_finger_joint1/2, each reporting half the aperture): the value we
# publish is the full aperture, so naming it after a finger joint would invite a 2x error.
GRIPPER_JOINT_NAME = 'fr3_gripper_width'

#: Health and latency scalars, published on /polyumi/diag/<name> as std_msgs/Float32 so they can be
#: watched as live timeseries in Foxglove's Plot panel (and recorded with `ros2 bag record`).
#:
#: Every one of these was already being computed for a log line. Logs are the wrong shape for them:
#: a number that matters as a *trend* scrolls past as a string, and the failure they describe is
#: usually gradual. `n_published_arm` is the one to put on the wall — it is the difference between
#: the arm moving and not, and it sat silently at 0 for a whole session before anyone noticed.
#:
#: One topic per scalar rather than a single custom message: this workspace has no rosidl package
#: (polyumi_pi_msgs is ament_python/protobuf), so a custom .msg would mean standing up a whole
#: ament_cmake interface package for a handful of floats. Foxglove's Plot panel takes one series
#: per path anyway. diagnostic_msgs/DiagnosticArray was the other candidate and is the wrong shape:
#: its values are strings, so it renders a status table rather than something plottable.
DIAG_METRICS = (
    'n_published_arm',
    'n_published_gripper',
    'n_stale_arm',
    'n_stale_gripper',
    'obs_age_s',
    'inference_latency_s',
    'image_age_s',
    'gripper_state_age_s',
)


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
        # 224 matches the model's shape_meta (camera0_rgb [3,224,224]). The resize below MUST
        # match the DP exporter's camera0_rgb contract exactly (RGB, 224x224, INTER_AREA, no
        # crop) — same pixels at train and inference. See ingest camera_preproc.resize_camera0_rgb
        # and docs/data-format.md ("camera0_rgb preprocessing contract").
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
        #
        # eef_frame is polyumi_tcp, NOT the stock fr3_hand_tcp: the policy's poses — both the
        # ones it observes here and the ones it predicts — live on the closed-fingertip midpoint
        # in optical axes (ingest step 5's `hand` body frame). Reading fr3_hand_tcp would hand it
        # a different physical point in a different axis convention. nuc/tcp_calib.py defines the
        # frame; the NUC's fr3_bringup.launch.py publishes it.
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('eef_frame', 'polyumi_tcp')
        # Look up the LATEST available EEF transform (tf2 time=0) instead of the latency-aligned
        # historical instant. For a stationary dry-run arm this sidesteps a laptop↔NUC clock skew
        # (TF stamps from another machine landing outside our buffer) at the cost of the
        # image/pose time-alignment — which only matters once the arm moves. Do NOT enable for
        # execution; fix the clock sync instead. Off by default.
        self.declare_parameter('tf_use_latest', False)
        # Motion execution (Phase 2). Off by default for safety: the node logs actions
        # but does NOT publish target poses unless execute_motion is explicitly enabled.
        # Planning params (group, velocity scaling) live on the NUC bridge, not here.
        self.declare_parameter('execute_motion', False)
        # Which executor the action chunk is aimed at. The two speak different message types on
        # different topics, so this decides what gets built and where it goes — see target_chunk.py.
        # Publishing both formats, as this node used to, meant the NUC alone decided which executor
        # acted, and a stack could drive MoveIt while looking like it was driving the servo.
        self.declare_parameter('wire', str(Wire.MULTIDOF))
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
        # plan one smooth path instead. The real default lives in config/inference.yaml, which
        # explains why the chunk has to span the whole observation->motion budget; this fallback
        # matches dummy_server's horizon, since that is what runs without the config loaded.
        self.declare_parameter('n_action_steps', 8)
        # Receding-horizon stride: run inference once per this many control ticks, not every
        # tick. The obs window still updates every tick (so it stays dt-spaced), but a new
        # chunk is only requested/published every steps_per_inference ticks — while the arm
        # executes the previous chunk. This is UMI's scheme (eval_real.py re-infers every
        # steps_per_inference steps, default 6) and it stops the per-tick POST/publish storm
        # that otherwise swamps the NUC bridge. 1 = infer every tick (the old behaviour).
        # Should be <= n_action_steps so each chunk covers the stride; a larger value just
        # means the arm runs out of fresh waypoints before the next chunk lands. The launch
        # default lives in config/inference.yaml (loaded via <param from>); this is the fallback.
        self.declare_parameter('steps_per_inference', 6)
        # Per-component system latencies (seconds), loaded from config/inference.yaml via the
        # inference launch file; that file documents each value's provenance. Measure them with
        # `ros2 run polyumi_ros2 latency_probe` (one mode per value) — procedures in
        # docs/calibration-instructions.md, "Latencies". gopro and proprio are consumed by
        # _lookup_agent_pos, arm_exec and gripper_exec by _n_stale_actions; finger_cam and
        # piezo_mic are declared but unused until the policy takes tactile input.
        self.declare_parameter('latency.gopro', 0.0)
        self.declare_parameter('latency.finger_cam', 0.0)
        self.declare_parameter('latency.piezo_mic', 0.0)
        self.declare_parameter('latency.proprio', 0.0)
        self.declare_parameter('latency.arm_exec', 0.0)
        # Delay from publishing a width to the fingers actually starting to move. The gripper's
        # counterpart to arm_exec, and separate from it because the two devices are genuinely
        # different speeds — the hand beat the arm by ~190 ms on first measurement. Each chunk is
        # truncated by its own device's value (see _post_and_act), which is UMI's split of
        # robot_action_latency vs gripper_action_latency.
        self.declare_parameter('latency.gripper_exec', 0.0)
        # Delay from the hand's true aperture to its measurement appearing on the joint-state
        # topic. Kept separate from latency.proprio because the gripper is a different device on a
        # different link. The topic jitters 24-100 ms, so the true value is not 0. This is the
        # OBSERVATION half, consumed by the width lookup; gripper_exec above is the action half.
        self.declare_parameter('latency.gripper', 0.0)
        # --- Gripper ---
        # Source for agent_pos[7]. The FR3 publishes each finger at HALF the aperture, so the two
        # positions are summed; see docs/crb-fr3-inference.md ("Gripper interface").
        self.declare_parameter('gripper_state_topic', '/fr3_gripper/joint_states')
        # If true, a tick with no gripper state is skipped (as a failed TF lookup is). Off by
        # default so setups without a hand — motion_only bringup, a bare arm — still run, feeding
        # the closed width with a throttled warning rather than stalling the whole loop.
        self.declare_parameter('require_gripper_state', False)
        # Reject a cached gripper sample older than this at lookup time. The gripper is the only
        # observation channel that can go stale *silently*: a dead camera trips max_image_age_s
        # and dead TF raises ExtrapolationException, but _gripper_width_at just keeps holding its
        # last sample forever, feeding the policy a frozen width with no complaint. 0.5s is ~5x
        # the worst observed publish interval (the topic jitters 24-100ms) so ordinary jitter
        # cannot trip it, and well short of the ~2.3s the buffer can hold. <= 0 disables the check.
        self.declare_parameter('max_gripper_age_s', 0.5)
        # Measured reachable aperture (`ros2 run polyumi_ros2 gripper_range_probe`), not the Franka
        # Hand's nominal range: max_width is published on no topic, so neither end is readable at
        # runtime. 0.0816 is the hand's own maximum here; the fingers limit neither end.
        #
        # gripper_min_width_m doubles as the policy->robot offset — policy width 0 is "fully
        # closed", which on the arm is this aperture. There is deliberately no separate
        # gripper_offset_m: it was always exactly -gripper_min_width_m, and two knobs for one
        # measurement could be set inconsistently. See gripper_map.
        self.declare_parameter('gripper_max_width_m', 0.0816)
        self.declare_parameter('gripper_min_width_m', 0.0)
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
        self._tf_use_latest = self.get_parameter('tf_use_latest').get_parameter_value().bool_value
        self._execute_motion = self.get_parameter('execute_motion').get_parameter_value().bool_value
        self._publish_preview = self.get_parameter('publish_preview').get_parameter_value().bool_value
        self._post_timeout_s = self.get_parameter('post_timeout_s').get_parameter_value().double_value
        self._n_action_steps = self.get_parameter('n_action_steps').get_parameter_value().integer_value
        self._steps_per_inference = self.get_parameter('steps_per_inference').get_parameter_value().integer_value
        control_hz = self.get_parameter('control_hz').get_parameter_value().double_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        gripper_topic = self.get_parameter('gripper_state_topic').get_parameter_value().string_value
        self._require_gripper_state = self.get_parameter('require_gripper_state').get_parameter_value().bool_value
        self._max_gripper_age_s = self.get_parameter('max_gripper_age_s').get_parameter_value().double_value
        self._gripper_max_width_m = self.get_parameter('gripper_max_width_m').get_parameter_value().double_value
        self._gripper_min_width_m = self.get_parameter('gripper_min_width_m').get_parameter_value().double_value
        self._latency = {
            'gopro': self.get_parameter('latency.gopro').get_parameter_value().double_value,
            'finger_cam': self.get_parameter('latency.finger_cam').get_parameter_value().double_value,
            'piezo_mic': self.get_parameter('latency.piezo_mic').get_parameter_value().double_value,
            'proprio': self.get_parameter('latency.proprio').get_parameter_value().double_value,
            'arm_exec': self.get_parameter('latency.arm_exec').get_parameter_value().double_value,
            'gripper_exec': self.get_parameter('latency.gripper_exec').get_parameter_value().double_value,
            'gripper': self.get_parameter('latency.gripper').get_parameter_value().double_value,
        }
        self._ee_pose_buffer_s = self.get_parameter('buffers.ee_pose_s').get_parameter_value().double_value
        self._validate_params(control_hz)

        # Observation age is no longer summed from constants — it's measured from the frame's
        # own stamp (see _n_stale_actions). latency.gopro still converts that stamp to a true
        # capture instant, and is the only delayed modality the policy consumes; once
        # finger_cam/piezo_mic feed it too, the capture instant becomes the oldest across them,
        # since an observation is only as fresh as its slowest stream.
        # Per-device execution delay. The arm and the hand are truncated independently, each by
        # its own value, so neither inherits the other's — see _post_and_act.
        self._latency_act = self._latency['arm_exec']
        self._latency_act_gripper = self._latency['gripper_exec']
        # Spacing between consecutive actions within a chunk. Assumes the policy's action
        # horizon runs at the observation/control rate (standard for UMI/diffusion policy);
        # if a model is ever trained at a different action rate this needs its own parameter.
        self._action_dt = 1.0 / control_hz

        # History buffers — each entry: (image_float32 [H,W,C], agent_pos [8])
        self._obs_buffer: deque = deque(maxlen=self._n_obs_steps)
        # Receding-horizon stride counter: inference runs on the tick where this is 0, then
        # every steps_per_inference ticks after. Kept in [0, steps_per_inference) so it never
        # grows. Starts at 0 so the first full-buffer tick infers immediately.
        self._inference_phase = 0
        self._latest_image: np.ndarray | None = None
        self._latest_image_stamp: rclpy.time.Time | None = None
        self._latest_image_lock = threading.Lock()
        # One-shot: the crop assumes the incoming frame is pillarboxed, and only a real frame can
        # confirm it. See _check_pillarbox_once.
        self._pillarbox_checked = False
        # Gripper aperture history, (stamp, width_m) oldest-first, for the same reason TF keeps a
        # buffer: the width must be sampled at the *frame's* capture instant, not at tick time.
        # tf2 does that interpolation for the pose; there's no equivalent for a plain topic, so we
        # keep a short ring and interpolate by hand (see _gripper_width_at). Sized to cover
        # ee_pose_s at the observed ~17 Hz with headroom, floored so a tiny buffer config can't
        # leave us with a single sample and no interval to interpolate over.
        self._gripper_buffer: deque = deque(maxlen=max(8, int(self._ee_pose_buffer_s * 40)))
        self._gripper_lock = threading.Lock()
        # Reject a cached frame older than this at tick time; a frame older than this means the
        # capture pipeline stalled. The auto default (max_image_age_s <= 0) is two camera periods
        # at the 60 Hz v4l2 rate, floored at half a control period so a slow tick doesn't trip it;
        # override the param for slower camera paths (see the param declaration).
        self._max_image_age_s = max_image_age_s if max_image_age_s > 0 else max(2.0 / 60.0, 0.5 / control_hz)

        # TF — cache_time sized from buffers.ee_pose_s so a pose from up to that far back can
        # still be looked up (needed to time-align with the delayed gopro frame; see
        # _lookup_agent_pos). tf2's buffer already interpolates (linear + slerp) between the
        # two nearest cached transforms for a historical lookup_transform() call, so there's
        # no need for a separate hand-rolled pose buffer.
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=self._ee_pose_buffer_s))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        # A lookup that has NEVER succeeded is a different fault from one that stopped working,
        # and the tf2 message for it ("frame does not exist") names the frame rather than the
        # cause. See _warn_no_tf_ever.
        self._tf_ever_ok = False

        # Motion execution (Phase 2). The MoveIt calls run in a bridge node ON THE NUC
        # (fr3_moveit_bridge), not here: the laptop (rmw_cyclonedds 4.x, Kilted) and NUC
        # (rmw 1.x, Humble) can exchange small messages but corrupt large MoveIt action
        # goals across the rmw-major boundary. So when execution is enabled we just publish
        # the target EEF pose chunk (PoseArray); the NUC bridge subscribes and plans+executes
        # the whole chunk as one Cartesian path via its local move_group.
        # The gripper rides a SEPARATE topic, not a field on the pose chunk: a PoseArray cannot
        # carry a width, and the Franka Hand is action-only (no ros2_control interface, libfranka
        # offers only blocking move/grasp), so it cannot be driven at the arm's cadence anyway.
        # fr3_gripper_bridge on the NUC deadbands and rate-limits it into Move/Grasp goals.
        #
        # The pose chunk goes to exactly ONE executor, chosen by `wire`. The MULTIDOF form carries
        # per-waypoint absolute times, which is what lets the NUC's 1 kHz interpolator splice chunks
        # without stopping; the PoseArray form carries no timing and leaves fr3_moveit_bridge to
        # re-time the chunk from arrival. It goes away with the MoveIt executor.
        self._target_pub = None
        self._gripper_pub = None
        if self._execute_motion:
            self._target_pub = TargetChunkPublisher(
                self,
                wire=self.get_parameter('wire').get_parameter_value().string_value,
                frame_id=self._base_frame,
                joint_name=self._eef_frame,
            )
            self._gripper_pub = self.create_publisher(JointTrajectory, '/polyumi/target_gripper', 10)

        # Viz-only preview publisher (always on when publish_preview). Shows every commanded chunk
        # in Foxglove/RViz without moving the arm: the NUC bridge subscribes only to the execution
        # topic /polyumi/target_poses, never this one.
        self._preview_pub = None
        self._gripper_preview_pub = None
        if self._publish_preview:
            self._preview_pub = self.create_publisher(PoseArray, '/polyumi/target_poses_preview', 10)
            self._gripper_preview_pub = self.create_publisher(JointTrajectory, '/polyumi/target_gripper_preview', 10)

        # Episode-start /reset. The server needs the episode-start EEF pose for
        # robot0_eef_rot_axis_angle_wrt_start; sent once on the first full-buffer tick. The reset
        # URL is derived from the predict URL's base so one param configures both endpoints.
        self._reset_url = self._url.split('/predict_cartesian')[0] + '/reset'
        self._episode_reset_done = False

        # Diagnostics. Always on: eight Float32s at the control rate is nothing next to the image
        # traffic already on the wire, and the failures these catch are exactly the ones you only
        # notice if the number was already being plotted.
        self._diag_pubs = {name: self.create_publisher(Float32, f'/polyumi/diag/{name}', 10) for name in DIAG_METRICS}

        # Subscribers.
        #
        # The gripper gets its OWN callback group. Left on the node's default group it shares one
        # MutuallyExclusiveCallbackGroup with the image subscription, and 60 Hz of 6 MB rgb8
        # deserialization starves it: measured on hardware, gripper callbacks gapped up to 1.4 s
        # and ~20% of samples were dropped outright (14.6 Hz delivered against a 17.9 Hz topic).
        # That is not merely a bad diagnostic — _gripper_width_at holds the nearest endpoint
        # outside its buffer span, so the policy was being handed a silently stale agent_pos[7],
        # and max_gripper_age_s tripped on most ticks. With its own group the callback tracks the
        # topic exactly (worst gap 122 ms against the topic's own 124 ms worst interval).
        #
        # Depth 1 on the image, not 10: only the newest frame is ever used (_image_cb overwrites
        # _latest_image), so a deeper queue just means working through stale frames after any
        # hiccup — burning CPU to make image_age_s worse.
        self.create_subscription(Image, image_topic, self._image_cb, 1)
        self.create_subscription(
            JointState,
            gripper_topic,
            self._gripper_cb,
            10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Control timer — exclusive callback group ensures only one tick (and its
        # blocking POST) runs at a time; an in-flight tick causes the next one to
        # be skipped rather than overlapping.
        self._tick_lock = threading.Lock()
        period = 1.0 / control_hz
        self.create_timer(period, self._control_tick, callback_group=MutuallyExclusiveCallbackGroup())

        # Throttle for "buffer not full" warning
        self._last_warn_t: rclpy.time.Time | None = None

        mode = 'EXECUTE (arm will move)' if self._execute_motion else 'log-only (no motion)'
        if self._target_pub is not None:
            mode += f' via {self._target_pub.wire} -> {self._target_pub.topic_name}'
        preview = 'on (/polyumi/target_poses_preview)' if self._publish_preview else 'off'
        self.get_logger().info(f'policy_client_node started — server: {self._url} — mode: {mode} — preview: {preview}')
        stride_interval = self._steps_per_inference * self._action_dt
        self.get_logger().info(
            f'receding-horizon stride — inference every {self._steps_per_inference} ticks '
            f'({stride_interval * 1e3:.0f}ms @ {control_hz:g}Hz), chunk n_action_steps={self._n_action_steps}'
        )
        latency_str = ' '.join(f'{name}={seconds}s' for name, seconds in self._latency.items())
        tf_mode = 'LATEST (clock-skew workaround; not time-aligned)' if self._tf_use_latest else 'time-aligned'
        self.get_logger().info(
            f'latency config — {latency_str} (ee_pose buffer: {self._ee_pose_buffer_s}s, '
            f'max_image_age: {self._max_image_age_s * 1e3:.0f}ms, tf lookup: {tf_mode})'
        )
        self.get_logger().info(
            f'latency budget — measured observation age (capture→response) + '
            f'act={self._latency_act}s arm / {self._latency_act_gripper}s gripper '
            f'vs action_dt={self._action_dt}s (each chunk truncated by its own device)'
        )
        gripper_missing = 'skip tick' if self._require_gripper_state else 'warn + use closed width'
        gripper_stale = 'skip tick' if self._require_gripper_state else 'warn + hold last width'
        max_age = f'{self._max_gripper_age_s}s' if self._max_gripper_age_s > 0 else 'disabled'
        self.get_logger().info(
            f'gripper — state: {gripper_topic} (missing: {gripper_missing}; '
            f'stale > {max_age}: {gripper_stale}), '
            f'aperture range [{self._gripper_min_width_m}, {self._gripper_max_width_m}]m '
            f'(the low end doubles as the policy->robot offset)'
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
        if self._steps_per_inference < 1:
            errors.append(f'steps_per_inference must be >= 1, got {self._steps_per_inference}')
        elif self._steps_per_inference > self._n_action_steps:
            # Not fatal — the arm just runs out of fresh waypoints before the next chunk — but
            # almost always a misconfiguration, so warn loudly rather than fail.
            self.get_logger().warn(
                f'steps_per_inference ({self._steps_per_inference}) > n_action_steps '
                f'({self._n_action_steps}): each chunk is shorter than the re-inference stride, '
                'so the arm will stall between chunks. Lower steps_per_inference or raise n_action_steps.'
            )
        if self._post_timeout_s <= 0:
            errors.append(f'post_timeout_s must be > 0, got {self._post_timeout_s}')
        for name, seconds in self._latency.items():
            if seconds < 0:
                errors.append(f'latency.{name} must be >= 0, got {seconds}')
        if self._gripper_max_width_m <= 0:
            errors.append(f'gripper_max_width_m must be > 0, got {self._gripper_max_width_m}')
        if self._gripper_min_width_m < 0:
            errors.append(f'gripper_min_width_m must be >= 0, got {self._gripper_min_width_m}')
        if self._gripper_min_width_m >= self._gripper_max_width_m:
            errors.append(
                f'gripper_min_width_m ({self._gripper_min_width_m}) must be < '
                f'gripper_max_width_m ({self._gripper_max_width_m})'
            )
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

    def _check_pillarbox_once(self, frame_rgb: np.ndarray) -> None:
        """
        On the first frame, confirm the crop is discarding black bars and not real image.

        Deferred to a frame rather than done at startup because there is nothing to inspect until
        one arrives. Logged, not fatal: a wrong crop still produces a running policy, and stopping
        the node mid-session is worse than telling the operator the observation is not what the
        policy trained on.
        """
        if self._pillarbox_checked:
            return
        self._pillarbox_checked = True
        bar = discarded_bar_intensity(frame_rgb)
        cropped = crop_to_source_aspect(frame_rgb)
        if bar <= MAX_BAR_INTENSITY:
            self.get_logger().info(
                f'camera crop: {frame_rgb.shape[1]}x{frame_rgb.shape[0]} → '
                f'{cropped.shape[1]}x{cropped.shape[0]}, discarded bars mean {bar:.1f}/255 — '
                'pillarbox as expected.'
            )
            return
        self.get_logger().error(
            f'camera crop is eating real image: the pixels dropped from this '
            f'{frame_rgb.shape[1]}x{frame_rgb.shape[0]} frame average {bar:.1f}/255, well above '
            f'{MAX_BAR_INTENSITY} — they are not a black pillarbox. The policy is being fed a '
            'narrower field of view than it trained on. Check the GoPro is in a 4:3 mode and the '
            'capture card is not rescaling. See docs/data-format.md ("Why the crop exists").'
        )

    def _image_cb(self, msg: Image) -> None:
        """Convert incoming ROS image to float32 numpy array and cache it with its stamp."""
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError(f'Unsupported image encoding {msg.encoding!r}; expected rgb8 or bgr8')
        if msg.step != msg.width * 3:
            raise ValueError(f'Unsupported row stride: step={msg.step} != width*3={msg.width * 3}')
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1].copy()  # BGR → RGB
        # Crop + INTER_AREA (the shared camera0_rgb contract) to match the DP exporter's
        # anti-aliased downscale; any mismatch here is train/inference skew. The crop drops the
        # GoPro HDMI pillarbox so this 16:9 capture squashes the same 4:3 field of view the 4:3
        # gopro.mp4 does. See camera_preproc.crop_to_source_aspect and docs/data-format.md.
        self._check_pillarbox_once(img)
        img = crop_to_source_aspect(img)
        resized = cv2.resize(img, (self._image_w, self._image_h), interpolation=CAMERA0_RGB_INTERPOLATION)
        float_img = resized.astype(np.float32) / 255.0
        with self._latest_image_lock:
            self._latest_image = float_img
            # Keep the frame's own stamp: the pose lookup must align to when THIS frame was
            # captured, not to when the control tick happens to run. The camera publishes at
            # 60 Hz while the tick runs at control_hz, so a cached frame is already up to one
            # camera period old before the tick even fires — and if the v4l2 pipeline stalls,
            # unboundedly older, with no way to notice. See _lookup_agent_pos.
            self._latest_image_stamp = rclpy.time.Time.from_msg(msg.header.stamp)

    def _gripper_cb(self, msg: JointState) -> None:
        """Cache the gripper aperture with its stamp, summing the two finger joints."""
        if len(msg.position) < 2:
            self._warn_throttled(
                f'Ignoring gripper state with {len(msg.position)} position(s); expected 2 (names: {list(msg.name)})'
            )
            return
        # Each FR3 finger reports HALF the aperture, so the full opening is their sum. Summing
        # rather than doubling position[0] keeps this honest if the fingers are ever asymmetric.
        width = float(msg.position[0] + msg.position[1])
        with self._gripper_lock:
            self._gripper_buffer.append((rclpy.time.Time.from_msg(msg.header.stamp), width))

    def _gripper_width_at(self, target: rclpy.time.Time | None) -> float | None:
        """
        Linearly interpolate the cached gripper aperture to ``target``, or None if unavailable.

        The same time-alignment discipline the TF lookup gets, done by hand because a plain topic
        has no tf2-style interpolating buffer. Outside the cached span we hold the nearest endpoint
        rather than extrapolating: the hand moves slowly relative to the ~60 ms sample interval, so
        a held value is a far smaller error than a linear extrapolation off the end would be.

        :param target: instant to sample the aperture at, or None for "latest available".
            Note this is deliberately **not** tf2's convention, where a zero ``Time()`` means
            latest — here a zero stamp is just an instant before every sample, and would return
            the *oldest* one. Callers with a tf2-style sentinel must translate it to None.
        :returns: aperture in metres, or None if no gripper state has been received at all.
        """
        with self._gripper_lock:
            samples = list(self._gripper_buffer)
        if not samples:
            return None
        if target is None:
            return samples[-1][1]

        t_ns = target.nanoseconds
        if t_ns <= samples[0][0].nanoseconds:
            return samples[0][1]
        if t_ns >= samples[-1][0].nanoseconds:
            return samples[-1][1]

        for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
            t0_ns, t1_ns = t0.nanoseconds, t1.nanoseconds
            if t0_ns <= t_ns <= t1_ns:
                if t1_ns == t0_ns:  # duplicate stamps — nothing to interpolate over
                    return w1
                alpha = (t_ns - t0_ns) / (t1_ns - t0_ns)
                return w0 + alpha * (w1 - w0)
        # Unreachable given the endpoint guards above, but a bad stamp ordering shouldn't crash
        # the control loop — fall back to the freshest sample.
        return samples[-1][1]

    def _diag(self, name: str, value: float) -> None:
        """
        Publish one diagnostic scalar on /polyumi/diag/<name>.

        :param name: must be in DIAG_METRICS; a typo is a KeyError at the call site rather than a
            topic nobody ever subscribes to and nobody notices is missing.
        """
        self._diag_pubs[name].publish(Float32(data=float(value)))

    def _newest_gripper_age_s(self) -> float | None:
        """
        Seconds between now and the freshest cached gripper sample, or None if there are none.

        Measured against the node clock rather than against the sample spacing, so a topic that
        stops entirely — the failure _gripper_width_at cannot see, since holding an endpoint looks
        identical to a slow publisher — grows this without bound.
        """
        with self._gripper_lock:
            if not self._gripper_buffer:
                return None
            newest = self._gripper_buffer[-1][0]
        return (self.get_clock().now() - newest).nanoseconds * 1e-9

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
            # Published before the guard, so a stalling capture pipeline shows up as a rising
            # trend rather than only as the warning it eventually trips.
            self._diag('image_age_s', image_age_s)
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

            # Receding-horizon stride: only the every-steps_per_inference-th tick actually runs
            # inference. The obs buffer was still appended above, so the next inference sees a
            # fresh dt-spaced window; we just don't re-POST/publish a chunk every tick (which
            # swamps the NUC bridge). Advance the phase AFTER deciding, kept in [0, stride).
            infer_now = self._inference_phase == 0
            self._inference_phase = (self._inference_phase + 1) % self._steps_per_inference
            if not infer_now:
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
        range. The gripper width gets the same treatment via _gripper_width_at, hand-rolled
        because a plain topic has no equivalent interpolating buffer.

        :param image_stamp: header stamp of the camera frame this pose will be paired with.
        :returns: the 8-vector agent_pos, or None if the tick should be skipped.
        """
        # tf2 time=0 means "latest available" — used for the dry-run clock-skew workaround.
        target_time = (
            rclpy.time.Time()
            if self._tf_use_latest
            else (image_stamp - Duration(seconds=self._latency['gopro']) + Duration(seconds=self._latency['proprio']))
        )
        try:
            tf = self._tf_buffer.lookup_transform(self._base_frame, self._eef_frame, target_time)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self._warn_throttled(f'TF lookup failed: {e}')
            self._warn_no_tf_ever()
            return None
        self._tf_ever_ok = True

        gripper_width = self._gripper_width_policy_units(image_stamp)
        if gripper_width is None:
            return None  # warning already logged inside

        t = tf.transform.translation
        r = tf.transform.rotation
        return np.array([t.x, t.y, t.z, r.x, r.y, r.z, r.w, gripper_width], dtype=np.float64)

    def _warn_no_tf_ever(self) -> None:
        """
        Report, once, that no arm TF has EVER arrived — which is an env fault, not a TF fault.

        The arm's frames come from the NUC, so "frame does not exist" from the very first tick
        means this process is not reaching the NUC at all. tf2 cannot say that; its message reads
        like the arm dropped out mid-run. The usual cause is a shell rc exporting its own
        ROS_DOMAIN_ID over the one tmux inherited, so the DDS env goes in the message.
        See docs/crb-fr3-inference.md.
        """
        if self._tf_ever_ok:
            return
        self.get_logger().error(
            f'No transform for {self._base_frame} has EVER arrived — this process is probably not '
            f'reaching the NUC at all, rather than having lost the arm mid-run. Check, in order: '
            f'(1) this shell sourced setup_franka_env.sh (an rc that exports its own '
            f'ROS_DOMAIN_ID overrides tmux and puts you on a private domain), '
            f'(2) NUC bringup is up (ros2 launch nuc/launch/fr3_bringup.launch.py). '
            f'This process: ROS_DOMAIN_ID={os.environ.get("ROS_DOMAIN_ID", "unset (0)")} '
            f'RMW_IMPLEMENTATION={os.environ.get("RMW_IMPLEMENTATION", "unset (default)")} '
            f'CYCLONEDDS_URI={os.environ.get("CYCLONEDDS_URI", "unset")}',
            once=True,
        )

    def _gripper_width_policy_units(self, image_stamp: rclpy.time.Time) -> float | None:
        """
        Sample the gripper aperture aligned to the frame, converted into the policy's units.

        Time-aligned exactly as the pose is, with latency.gripper standing in for
        latency.proprio — the hand is a separate device reporting on its own topic at its own
        rate, which is why UMI also keeps the two constants apart.

        :param image_stamp: header stamp of the camera frame this observation belongs to.
        :returns: width in policy units (finger-tag separation), or None if the tick should be
            skipped because require_gripper_state is set and no usable state is available.
        """
        # None (not a zero Time()) is this buffer's "latest available" sentinel — tf2's zero-stamp
        # convention does not carry over, and passing it through would return the OLDEST sample.
        target_time = (
            None
            if self._tf_use_latest
            else (image_stamp - Duration(seconds=self._latency['gopro']) + Duration(seconds=self._latency['gripper']))
        )
        width = self._gripper_width_at(target_time)
        if width is None:
            if self._require_gripper_state:
                self._warn_throttled(
                    'Dropped control tick: no gripper state received yet (require_gripper_state is set)'
                )
                return None
            # Substituting the closed width is a lie to the policy, but a survivable one — it keeps
            # arm-only bringup (motion_only, no hand) working. The startup banner says which mode
            # is active so this isn't silent.
            self._warn_throttled(
                'No gripper state received yet; substituting closed width for agent_pos[7]. '
                'Set require_gripper_state:=true to skip these ticks instead.'
            )
            # Fully closed is 0.0 in policy units by definition — the exporter subtracted the
            # closed width, so the scale starts at "shut". Converting a robot-side 0.0 here would
            # be wrong for fingers whose closed aperture is non-zero: that aperture is not merely
            # closed, it is narrower than the hand can physically go.
            return 0.0

        # A topic that stops publishing is invisible to _gripper_width_at — it just keeps holding
        # its newest sample — so the age is checked explicitly, as the camera path does. Skipped
        # under tf_use_latest, which exists precisely because the stamps are known to be skewed
        # against this clock and would false-trip it.
        age_s = self._newest_gripper_age_s()
        if age_s is not None:
            # Same reasoning as image_age_s: published whether or not it trips the limit, since a
            # topic slowing down is the interesting part and it only ever trips once.
            self._diag('gripper_state_age_s', age_s)
        if (
            not self._tf_use_latest
            and self._max_gripper_age_s > 0
            and age_s is not None
            and age_s > self._max_gripper_age_s
        ):
            if self._require_gripper_state:
                self._warn_throttled(
                    f'Dropped control tick: newest gripper state is {age_s * 1e3:.0f} ms old '
                    f'(limit {self._max_gripper_age_s * 1e3:.0f} ms) — has the gripper topic died?'
                )
                return None
            # Holding the last known width beats substituting closed: if the hand stopped
            # reporting mid-grasp, "closed" is a bigger lie than "still where we last saw it".
            self._warn_throttled(
                f'Newest gripper state is {age_s * 1e3:.0f} ms old '
                f'(limit {self._max_gripper_age_s * 1e3:.0f} ms) — has the gripper topic died? '
                'Holding the last known width for agent_pos[7].'
            )
        return robot_to_policy_width(width, self._gripper_min_width_m)

    def _n_stale_actions(self, t_obs: rclpy.time.Time, latency_act: float) -> int:
        """
        Count the leading actions in a chunk that are already in the past by execution time.

        Action i in a chunk is the policy's target for t_obs + i * action_dt, where t_obs is
        the instant the observation was captured. Everything between that instant and the device
        actually moving makes the leading actions stale: the frame's transit through the
        capture pipeline, the tick's own serialization work, the server's round trip, and
        ``latency_act`` before the device starts moving. Actions whose target instant falls inside
        that window have already elapsed — executing them would drag the device backwards through
        the trajectory — so skip to the first one still in the future.

        Called after the server responds, so ``now() - t_obs`` measures every delay up to this
        point directly (capture pipeline + tick + inference round trip) instead of summing
        assumed constants for them; only ``latency_act`` is still in the future and must be added.

        Clamped at 0: if t_obs somehow lands in the future (clock skew between the camera
        driver's stamps and this node), a negative count would slice from the *end* of the
        chunk — `actions[-3:]` keeps the last three, jumping the device ahead to the far-future
        tail of the trajectory instead of dropping anything. No upper clamp is needed: a count
        at or past the chunk length yields an empty slice, which the caller already reports.

        :param t_obs: instant the observation was captured (image stamp - latency.gopro).
        :param latency_act: this device's publish-to-motion delay, in seconds. Passed rather than
            read from self because the arm and the hand get different values — see _post_and_act.
        :return: number of actions to drop from the front of the chunk, >= 0.
        """
        elapsed_since_obs = (self.get_clock().now() - t_obs).nanoseconds * 1e-9
        total_latency = elapsed_since_obs + latency_act
        return max(0, math.ceil(total_latency / self._action_dt))

    def _http_post_json(self, url: str, payload: dict) -> dict | None:
        """POST payload as JSON to url and return the parsed response, or None on failure (logged)."""
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
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
            self._warn_throttled('episode /reset failed; server will approximate wrt_start with the current pose')

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
        if self._gripper_preview_pub is not None:
            self._gripper_preview_pub.publish(self._actions_to_gripper_trajectory(actions))

        # Drop the leading actions that refer to instants already elapsed by the time each device
        # can act on them, so execution starts from the first still-future waypoint.
        #
        # The two devices are truncated INDEPENDENTLY, because they are genuinely different
        # speeds: the hand starts moving ~190 ms before the arm does. A single shared slice would
        # force the faster device to inherit the slower one's lead and act that much too early,
        # which is what fr3_gripper_bridge's (now removed) gripper_lead_steps existed to claw
        # back. This is UMI's split — robot_action_latency vs gripper_action_latency, each
        # subtracted per device — reached through slicing rather than absolute waypoint times,
        # since a PoseArray carries no timing. Note _actions_to_gripper_trajectory recomputes
        # time_from_start relative to whatever slice it is handed, so the two stay self-consistent.
        n_received = len(actions)
        n_stale_arm = self._n_stale_actions(t_obs, self._latency_act)
        n_stale_grip = self._n_stale_actions(t_obs, self._latency_act_gripper)
        arm_actions = actions[n_stale_arm:]
        grip_actions = actions[n_stale_grip:]

        # Published before the all-stale check, so the zero is visible: "nothing was commanded" is
        # the single most useful thing on the plot and it is exactly the case that returns early.
        age_s = (self.get_clock().now() - t_obs).nanoseconds * 1e-9
        self._diag('obs_age_s', age_s)
        self._diag('inference_latency_s', latency_inference)
        self._diag('n_stale_arm', n_stale_arm)
        self._diag('n_stale_gripper', n_stale_grip)
        self._diag('n_published_arm', len(arm_actions))
        self._diag('n_published_gripper', len(grip_actions))

        if not arm_actions and not grip_actions:
            self._warn_throttled(
                f'Whole action chunk stale for both devices: dropped all {n_received} actions '
                f'(observation is {age_s:.3f}s old, of which inference={latency_inference:.3f}s; '
                f'plus act={self._latency_act:.3f}s arm / {self._latency_act_gripper:.3f}s gripper '
                f'exceeds chunk span {n_received * self._action_dt:.3f}s). '
                'Raise n_action_steps or reduce latency.'
            )
            return

        # Log against whichever device still has waypoints; the faster one outlives the other.
        first = (arm_actions or grip_actions)[0]
        # Log the width in both spaces: policy units are what the model emitted, robot units are
        # what the hand will be commanded. A surprising gap between them is the offset being wrong.
        grip_robot = policy_to_robot_width(float(first[7]), self._gripper_min_width_m, self._gripper_max_width_m)
        self.get_logger().info(
            f'action chunk n={n_received} (dropped {n_stale_arm} arm / {n_stale_grip} gripper, '
            f'inference={latency_inference * 1000:.0f}ms) first: x={first[0]:.4f} y={first[1]:.4f} '
            f'z={first[2]:.4f} grip={first[7]:.3f}→{grip_robot:.3f}m'
        )

        # Phase 2: publish the whole action chunk for the NUC bridge to plan+execute as one
        # Cartesian path (receding-horizon control). Non-blocking (unlike a direct MoveIt
        # call): the NUC bridge does its own skip-while-busy, so at worst it drops chunks
        # that arrive mid-motion. The gripper half goes out on its own topic and its own slice.
        # Each is published only if it still has waypoints, so a chunk too stale for the arm can
        # still drive the hand rather than stalling both.
        if self._target_pub is not None and arm_actions:
            # Anchored at t_obs minus latency.arm_exec, so every waypoint is commanded that far
            # ahead of when it should be reached — UMI's per-waypoint `target_time -
            # robot_action_latency` (exec_actions in bimanual_umi_env.py), folded into the anchor
            # because the offset is the same for every waypoint. first_index is the index in the
            # ORIGINAL chunk: numbering the survivors of the stale-drop from zero would slide the
            # whole timeline earlier. PoseArray ignores both, carrying no timing at all.
            if self._target_pub.get_subscription_count() == 0:
                self._warn_throttled(
                    f'Nothing is subscribed to {self._target_pub.topic_name}; the arm will not '
                    f'move. Needs: {self._target_pub.wire.consumer}'
                )
            self._target_pub.publish(
                [self._action_to_pose(action) for action in arm_actions],
                dt=self._action_dt,
                first_index=n_stale_arm,
                stamp=(t_obs - Duration(seconds=self._latency_act)).to_msg(),
            )
        if self._gripper_pub is not None and grip_actions:
            self._gripper_pub.publish(self._actions_to_gripper_trajectory(grip_actions))

    @staticmethod
    def _action_to_pose(action) -> Pose:
        """Convert one 8-vector action [x,y,z,qx,qy,qz,qw,grip] to a Pose, dropping the width."""
        pose = Pose()
        pose.position.x = float(action[0])
        pose.position.y = float(action[1])
        pose.position.z = float(action[2])
        pose.orientation.x = float(action[3])
        pose.orientation.y = float(action[4])
        pose.orientation.z = float(action[5])
        pose.orientation.w = float(action[6])
        return pose

    def _actions_to_pose_array(self, actions) -> PoseArray:
        """Build a PoseArray in base_frame from a list of 8-vector actions [x,y,z,qx,qy,qz,qw,grip]."""
        return pose_array(
            [self._action_to_pose(action) for action in actions],
            frame_id=self._base_frame,
            stamp=self.get_clock().now().to_msg(),
        )

    def _actions_to_gripper_trajectory(self, actions) -> JointTrajectory:
        """
        Build the gripper half of an action chunk as a timed single-DOF trajectory.

        Carries per-point ``time_from_start`` so the NUC bridge can pick a lead waypoint and derive
        a move speed from it, rather than commanding every width at one fixed speed. The widths are
        converted to robot jaw aperture here so the bridge stays free of calibration — see
        polyumi_ros2.gripper_map.

        Times run from the SLICE index, deliberately unlike the arm chunk, which is numbered from
        its pre-slice index: this consumer reads time_from_start as a relative shape to derive a
        speed, not as an absolute schedule, so shifting it changes nothing.

        :param actions: 8-vector actions [x,y,z,qx,qy,qz,qw,grip], grip in policy units.
        """
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.joint_names = [GRIPPER_JOINT_NAME]
        for i, action in enumerate(actions):
            point = JointTrajectoryPoint()
            point.positions = [
                policy_to_robot_width(float(action[7]), self._gripper_min_width_m, self._gripper_max_width_m)
            ]
            point.time_from_start = Duration(seconds=i * self._action_dt).to_msg()
            msg.points.append(point)
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

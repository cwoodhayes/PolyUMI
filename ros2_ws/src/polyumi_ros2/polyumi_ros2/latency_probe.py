#!/usr/bin/env python3
"""
Measure the system latencies ``config/inference.yaml`` guesses at, one mode per quantity.

The inference loop compensates latency in three places — the TF pose lookup, the gripper-width
lookup, and ``t_obs`` for action-chunk truncation — and until now every constant feeding them was a
plausible guess. This probe replaces the guesses with measurements, porting the two estimators
upstream UMI uses (``scripts/calibrate_uvc_camera_latency.py`` and
``scripts/calibrate_{robot,gripper}_latency.py``).

Only one constant in the whole observation->action budget actually needs calibrating, which is why
this is smaller than it looks::

    photon --(latency.gopro)--> header.stamp --(measured live)--> response --(*_exec)--> motion
            mode:=camera                       policy_client_node          mode:=arm / :=gripper
                                               already measures this

The trailing term is per device: the arm and the hand each have their own ``latency.*_exec`` and
``policy_client_node`` truncates their chunks separately, so the two are measured and configured
independently. Nothing here has to be reconciled against the other device.

``header.stamp`` is the earliest instant the laptop has any handle on, so ``mode:=camera`` measures
photon->stamp and everything downstream of the stamp — the YUYV convert, tick phasing, the POST, the
network, the server's own inference time — is already measured live by ``_n_stale_actions``, which
runs *after* the response lands and uses ``now() - t_obs``. There is deliberately no round-trip
constant to calibrate; a config value would be strictly worse than that live measurement.

Modes
-----

``mode:=camera``
    ``latency.gopro``. Displays a QR code encoding the current time, which the GoPro films off the
    screen; the lag is the mean of (frame stamp - encoded time), less the time the screen took to
    render. Needs no robot.

``mode:=arm``
    ``latency.arm_exec``. Chirps the commanded EEF pose along one axis and cross-correlates it
    against the ``polyumi_tcp`` TF. **Moves the arm.**

``mode:=gripper``
    ``latency.gripper_exec``, the hand's command->motion delay, plus the
    ``/fr3_gripper/joint_states`` publish interval that sets ``latency.gripper``.
    **Moves the fingers.**

Usage (laptop, after ``source setup_franka_env.sh``)::

    # 1. camera — point the GoPro at the laptop screen, filling the frame, and focused
    ros2 launch polyumi_ros2 stream_demo.launch.xml motion_only:=true
    ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=camera

    # 2. arm — NUC: ros2 launch nuc/launch/fr3_inference.launch.py execute_arm:=true
    ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"   # a roomy pose; edge poses fail to plan
    ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=arm

    # 3. gripper — NUC: ... execute_gripper:=true, and nothing between the fingers
    ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=gripper

Each mode prints the number, its quality, and the exact line to paste, then saves the raw series to
``.npz`` so a marginal run can be re-judged without booking the hardware again. Pass ``-p
plot:=false`` on a headless box (``mode:=camera`` needs a screen regardless — it is the light
source).

**Do not run the arm or gripper modes while policy_client_node is up.** They publish to the same
topics and the bridges act on whichever chunk arrived last.
"""

import threading
import time

import cv2
from geometry_msgs.msg import Pose, PoseArray
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import scipy.signal as ss
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from polyumi_ros2.latency_util import get_latency

MODES = ('camera', 'arm', 'gripper')

#: Chirp band, in Hz. Bandwidth is what makes the correlation peak sharp (see latency_util), so
#: this sits as high as the arm can actually follow through MoveIt's plan-then-execute cadence.
#: Driving it faster does not sharpen the peak, it just stops exciting the plant. Only the arm
#: chirps; the gripper is measured by step response instead (see _run_gripper).
CHIRP_BAND_HZ = {'arm': (0.05, 0.4)}

#: Arm command rate, deliberately slow: `fr3_moveit_bridge` plans and executes each chunk
#: synchronously and drops any that arrive mid-flight, so commanding at the 10 Hz control rate
#: would discard most of them. Dropping *some* is harmless — the arm still traverses the same
#: smooth chirp, and it is the chirp we correlate against — but it is noise we can avoid.
COMMAND_HZ = {'arm': 4.0}

#: Joint name on /polyumi/target_gripper, matching policy_client_node and gripper_range_probe.
GRIPPER_JOINT_NAME = 'fr3_gripper_width'

#: Window the gripper must hold still over before a step is timed, and the spread that counts as
#: still. Longer than the bridge's 0.25 s command period, so a rep cannot start while the previous
#: goal is still in flight and be timed against a hand that was already moving.
GRIPPER_SETTLE_S = 0.5
GRIPPER_STILL_TOL_M = 0.0005

#: DDS discovery is asynchronous; a command published before the endpoints pair goes nowhere,
#: silently.
SUBSCRIBER_TIMEOUT_S = 10.0

#: How often to sample TF during the arm run. Faster than the ~30 Hz the arm state actually
#: arrives at, so the sampler is never the limiting factor; duplicate stamps are dropped.
TF_SAMPLE_HZ = 100.0

#: QR refresh rate. Each code must persist for at least a couple of 60 fps frames to be decodable,
#: and every code that appears is one usable measurement.
QR_HZ = 20.0

#: Cap on buffered camera frames, bounded so a long run cannot exhaust RAM: single-channel 1080p is
#: ~2 MB each, so this is ~400 MB. At the camera's 60 fps that is ~3.3 s, which at QR_HZ yields ~65
#: distinct codes — comfortably more measurements than the mean needs. The camera run therefore
#: stops as soon as the buffer fills, and duration_s is an upper bound rather than the run length.
MAX_BUFFERED_FRAMES = 200

#: Below this the correlation peak is noise rather than a match, and the lag means nothing.
MIN_PEAK_CORR = 0.5

#: Above this the peak is too broad to localise the lag; the excitation was too slow or too small.
#: Generous, because the arm mode cannot be driven broadband and legitimately lands near here.
MAX_PEAK_WIDTH_S = 0.4


#: Quiet zone around the QR, in modules. The spec requires 4; without it no decoder finds the
#: finder patterns, and cv2's encoder emits the bare symbol.
QR_QUIET_ZONE_MODULES = 4

#: On-screen size of the rendered code. Nearest-neighbour up from ~25 modules keeps the edges hard,
#: which is what survives being filmed and rescaled.
QR_RENDER_PX = 720


def render_qr(encoder, text: str) -> np.ndarray:
    """
    Encode ``text`` as a QR image big enough to film, quiet zone included.

    Uses OpenCV's own encoder rather than the ``qrcode`` package upstream UMI reaches for: cv2 is
    already a hard dependency of this package, and this leaves the probe with no new one at all.

    :param encoder: a ``cv2.QRCodeEncoder``.
    :param text: payload; the probe encodes a timestamp.
    :returns: a square uint8 grayscale image.
    """
    symbol = np.pad(encoder.encode(text), QR_QUIET_ZONE_MODULES, constant_values=255)
    return cv2.resize(symbol, (QR_RENDER_PX, QR_RENDER_PX), interpolation=cv2.INTER_NEAREST)


def decode_qr(detector, gray: np.ndarray) -> str:
    """
    Read a QR out of one filmed frame, returning '' when there is none.

    Deliberately ``detectAndDecode`` and **not** ``detectAndDecodeCurved``, which is what upstream
    UMI's camera script calls. On OpenCV 4.6 the curved variant decodes nothing on a flat,
    undistorted code — including a synthetic one straight out of the encoder — so porting it
    verbatim would have produced zero readings and looked like a bad rig. Measured across eight
    simulated capture conditions (blur, perspective, noise, scale) the plain detector read all
    eight and the curved one four. A monitor is flat; there is nothing curved to correct for.

    :param detector: a ``cv2.QRCodeDetector``.
    :param gray: single-channel frame.
    :returns: the decoded payload, or '' if the frame holds no readable code.
    """
    try:
        text, _, _ = detector.detectAndDecode(gray)
    except cv2.error:
        return ''
    return text


def first_stamp_per_code(decoded) -> dict[float, float]:
    """
    Reduce decoded frames to one frame stamp per distinct QR code — the earliest.

    One code stays on screen across several 60 fps frames, so a code decoded from N frames yields N
    stamps for a single display instant. Averaging all of them would bias every offset upward by
    roughly half a QR period, which at 20 Hz is 25 ms — comparable to the differences we are trying
    to resolve. Keeping the first is upstream UMI's behaviour and the reason its camera script does
    a second, offline decoding pass.

    :param decoded: iterable of ``(frame_stamp_s, decoded_text)``; unparseable text is skipped.
    :returns: ``{encoded_time_s: earliest frame stamp that saw it}``.
    """
    first: dict[float, float] = {}
    for stamp_s, text in decoded:
        if not text:
            continue
        try:
            qr_time = float(text)
        except ValueError:
            continue
        # Frames arrive in order, but a dropped-and-resent frame or a reordered buffer would
        # otherwise let a later stamp win.
        if qr_time not in first or stamp_s < first[qr_time]:
            first[qr_time] = stamp_s
    return first


class LatencyProbe(Node):
    """Drive one excitation, record the response, and report the lag between them."""

    def __init__(self, **kwargs):
        """
        Declare params for the selected mode and wire up only that mode's pubs/subs.

        :param kwargs: forwarded to rclpy's Node — notably ``parameter_overrides``, matching the
            other probes so this can be constructed under test without a launch file.
        :raises ValueError: on an unknown mode or a non-physical parameter value.
        """
        super().__init__('latency_probe', **kwargs)

        self.declare_parameter('mode', 'camera')
        self._mode = self.get_parameter('mode').get_parameter_value().string_value
        if self._mode not in MODES:
            self.destroy_node()
            raise ValueError(f"mode must be one of {MODES}, got '{self._mode}'")

        self.declare_parameter('duration_s', 20.0)
        self.declare_parameter('plot', True)
        self.declare_parameter('output_npz', '')
        # Only used to phrase the arm result against the control period; the gripper result is a
        # latency in seconds and needs no conversion. Must match policy_client_node's control rate.
        self.declare_parameter('action_dt', 0.1)

        self._duration_s = self.get_parameter('duration_s').get_parameter_value().double_value
        self._plot = self.get_parameter('plot').get_parameter_value().bool_value
        self._output_npz = self.get_parameter('output_npz').get_parameter_value().string_value
        self._action_dt = self.get_parameter('action_dt').get_parameter_value().double_value

        errors = []
        if self._duration_s <= 1.0:
            errors.append(f'duration_s must be > 1.0, got {self._duration_s}')
        if self._action_dt <= 0:
            errors.append(f'action_dt must be > 0, got {self._action_dt}')

        if self._mode == 'camera':
            errors += self._init_camera()
        else:
            errors += self._init_motion()

        if errors:
            # Node.__init__ has already registered us with the rclpy context, so tear that down
            # before raising, or every rejected construction leaves an orphan node behind.
            self.destroy_node()
            raise ValueError('; '.join(errors))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_camera(self) -> list[str]:
        """Declare camera-mode params and subscribe to the image topic."""
        self.declare_parameter('image_topic', '/gopro/image_raw')
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self._lock = threading.Lock()
        #: (stamp_s, arrival_s, grayscale frame) — arrival is kept for the max_image_age_s
        #: diagnostic, which is a different quantity from latency.gopro (see _report_camera).
        self._frames: list[tuple[float, float, np.ndarray]] = []
        self.create_subscription(Image, image_topic, self._on_image, 10)
        return []

    def _init_motion(self) -> list[str]:
        """Declare arm/gripper params and wire the command publisher and state source."""
        errors = []
        self._lock = threading.Lock()
        if self._mode == 'arm':
            f0, f1 = CHIRP_BAND_HZ['arm']
            self.declare_parameter('chirp_f0_hz', f0)
            self.declare_parameter('chirp_f1_hz', f1)
            self.declare_parameter('command_hz', COMMAND_HZ['arm'])
            self._f0 = self.get_parameter('chirp_f0_hz').get_parameter_value().double_value
            self._f1 = self.get_parameter('chirp_f1_hz').get_parameter_value().double_value
            self._command_hz = self.get_parameter('command_hz').get_parameter_value().double_value
            if not 0 < self._f0 < self._f1:
                errors.append(f'need 0 < chirp_f0_hz < chirp_f1_hz, got {self._f0} and {self._f1}')
            if self._command_hz <= 0:
                errors.append(f'command_hz must be > 0, got {self._command_hz}')
            # Below 2 samples per cycle the commanded chirp aliases, and we would correlate the
            # arm's motion against a signal that was never really commanded.
            elif self._f1 >= self._command_hz / 2:
                errors.append(
                    f'chirp_f1_hz ({self._f1}) must be below the Nyquist of command_hz '
                    f'({self._command_hz / 2}), or the commanded sweep aliases'
                )
            self.declare_parameter('target_topic', '/polyumi/target_poses')
            self.declare_parameter('base_frame', 'fr3_link0')
            self.declare_parameter('eef_frame', 'polyumi_tcp')
            self.declare_parameter('amplitude_m', 0.03)
            # Lateral by default: a vertical sweep loads and unloads gravity asymmetrically, so
            # the arm's lag differs between the up and down halves and smears the peak.
            self.declare_parameter('axis', 'y')
            self._amplitude = self.get_parameter('amplitude_m').get_parameter_value().double_value
            self._axis = self.get_parameter('axis').get_parameter_value().string_value
            self._base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
            self._eef_frame = self.get_parameter('eef_frame').get_parameter_value().string_value
            if self._axis not in ('x', 'y', 'z'):
                errors.append(f"axis must be x, y or z, got '{self._axis}'")
            if self._amplitude <= 0:
                errors.append(f'amplitude_m must be > 0, got {self._amplitude}')
            topic = self.get_parameter('target_topic').get_parameter_value().string_value
            self._pub = self.create_publisher(PoseArray, topic, 10)
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            #: (stamp_s, position on the swept axis)
            self._actual: list[tuple[float, float]] = []
        else:
            self.declare_parameter('target_topic', '/polyumi/target_gripper')
            self.declare_parameter('state_topic', '/fr3_gripper/joint_states')
            # Step size, comfortably past the bridge's 0.005 m width_deadband_m so the command is
            # never swallowed, and centred mid-stroke so neither end hits a hard stop.
            self.declare_parameter('step_m', 0.03)
            self.declare_parameter('center_width_m', 0.04)
            self.declare_parameter('reps', 8)
            # Onset threshold. Well above the encoder's resting noise (the probe prints what it
            # actually measured, so you can check) and well below the step, so the trigger fires on
            # the start of the move rather than partway through it.
            self.declare_parameter('onset_threshold_m', 0.001)
            self.declare_parameter('settle_timeout_s', 10.0)
            self._step = self.get_parameter('step_m').get_parameter_value().double_value
            self._center = self.get_parameter('center_width_m').get_parameter_value().double_value
            self._reps = self.get_parameter('reps').get_parameter_value().integer_value
            self._onset_threshold = (
                self.get_parameter('onset_threshold_m').get_parameter_value().double_value
            )
            self._settle_timeout_s = (
                self.get_parameter('settle_timeout_s').get_parameter_value().double_value
            )
            if self._step <= 0.005:
                errors.append(f"step_m must exceed the bridge's 0.005 m deadband, got {self._step}")
            if self._center - self._step / 2 < 0:
                errors.append(
                    f'center_width_m ({self._center}) - step_m/2 ({self._step / 2}) is negative; '
                    'the step would command the hand past shut'
                )
            if self._reps < 3:
                errors.append(f'reps must be >= 3 to take a median, got {self._reps}')
            if not 0 < self._onset_threshold < self._step / 2:
                errors.append(
                    f'onset_threshold_m ({self._onset_threshold}) must be positive and well under '
                    f'half the step ({self._step / 2})'
                )
            topic = self.get_parameter('target_topic').get_parameter_value().string_value
            state_topic = self.get_parameter('state_topic').get_parameter_value().string_value
            self._pub = self.create_publisher(JointTrajectory, topic, 10)
            self.create_subscription(JointState, state_topic, self._on_gripper_state, 10)
            #: (stamp_s, aperture)
            self._actual: list[tuple[float, float]] = []
        return errors

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        """Buffer a single-channel copy of the frame with both its stamp and its arrival time."""
        if self.frames_full():
            return
        # Same encoding contract as policy_client_node: v4l2_camera converts to RGB before
        # publishing, which is where the ~200 ms convert cost lives.
        if msg.encoding not in ('rgb8', 'bgr8'):
            self.get_logger().warning(f'Unhandled image encoding {msg.encoding}; skipping frame')
            return
        # Green channel rather than a real luma conversion: the code is black on white, so every
        # channel carries it, and keeping one is what makes buffering a whole run affordable
        # (~2 MB per frame instead of 6). The channel is the same in rgb8 and bgr8.
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)[:, :, 1].copy()
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._frames.append((stamp_s, self._now(), gray))

    def frames_full(self) -> bool:
        """Whether the camera buffer has all the frames it can hold, so the run can stop early."""
        with self._lock:
            return len(self._frames) >= MAX_BUFFERED_FRAMES

    def _on_gripper_state(self, msg: JointState) -> None:
        """Record the aperture against its own stamp; each FR3 finger reports half of it."""
        if len(msg.position) < 2:
            return
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._actual.append((stamp_s, float(msg.position[0] + msg.position[1])))

    def _now(self) -> float:
        """
        Seconds on the node clock, i.e. the LAPTOP's clock.

        ``mode:=camera`` is self-contained on this clock: the QR payload, the frame stamp
        (``v4l2_camera`` runs here) and the arrival time all come from it, so nothing crosses a
        clock domain and the answer is immune to any skew.

        The arm and gripper modes are NOT. Their commands are stamped here while the responses —
        ``/fr3_gripper/joint_states`` headers, TF from the arm — are stamped on the NUC, so the
        laptop<->NUC offset lands in the result one-for-one. That is fine only because the two are
        chrony-disciplined to sub-ms over the 10.0.0.x link (see CLAUDE.md, "Clock sync"); if that
        has drifted, these modes silently measure the drift. ``ssh jailfranka chronyc sources``
        before believing a surprising number.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Execute the selected mode and report. Returns a process exit code."""
        if self._mode == 'camera':
            return self._run_camera()
        if not self._wait_for_subscriber():
            return 1
        return self._run_arm() if self._mode == 'arm' else self._run_gripper()

    def _wait_for_subscriber(self) -> bool:
        """Block until a bridge has matched our command topic, so commands are not published into a void."""
        deadline = time.monotonic() + SUBSCRIBER_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._pub.get_subscription_count() > 0:
                return True
            time.sleep(0.1)
        self.get_logger().error(
            f'Nothing is subscribed to {self._pub.topic_name} after {SUBSCRIBER_TIMEOUT_S}s. Is the '
            'NUC bridge running, with its execute flag set? Without it every command is a silent no-op.'
        )
        return False

    def _chirp(self, elapsed: np.ndarray | float) -> np.ndarray | float:
        """
        Unit-amplitude linear frequency sweep over the run.

        A sweep rather than UMI's fixed sine because bandwidth is what localises the correlation
        peak — the same reason the audio time sync uses a chirp. See :mod:`polyumi_ros2.latency_util`.
        """
        t = np.clip(elapsed, 0.0, self._duration_s)
        return ss.chirp(t, f0=self._f0, f1=self._f1, t1=self._duration_s, method='linear')

    def _run_camera(self) -> int:
        """Film a clock off the screen and report how far behind the frame stamps run."""
        encoder = cv2.QRCodeEncoder.create()
        window = 'PolyUMI latency probe — point the GoPro here'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.get_logger().info(
            f'Displaying a timestamp QR for {self._duration_s:g}s. Fill the GoPro frame with the '
            'screen, in focus. Press q to stop early.'
        )

        render_overheads = []
        period_ms = max(1, int(1000 / QR_HZ))
        end = self._now() + self._duration_s
        key = -1
        while self._now() < end and key & 0xFF != ord('q'):
            t_sample = self._now()
            cv2.imshow(window, render_qr(encoder, f'{t_sample:.6f}'))
            # TWO waitKeys, and the split is the whole point. imshow only queues; waitKey is what
            # pumps highgui's event loop and actually paints, so the code is on screen once
            # waitKey(1) returns — that, and only that, is the delay we caused and must subtract.
            # The rest of the QR period is paced by the SECOND waitKey, outside the measurement.
            # Measuring across the pacing wait (as this did until 2026-08-11) folds a whole
            # 1/QR_HZ — 50 ms, a third of latency.gopro itself — into the correction and
            # under-reports the answer by that much.
            key = cv2.waitKey(1)
            render_overheads.append(self._now() - t_sample)
            # Stop as soon as we have all the frames we can hold, rather than displaying to a
            # buffer that is silently discarding everything.
            if self.frames_full():
                self.get_logger().info(f'Buffered {MAX_BUFFERED_FRAMES} frames; stopping early.')
                break
            remaining_ms = period_ms - int((self._now() - t_sample) * 1e3)
            if remaining_ms > 0:
                key = cv2.waitKey(remaining_ms)
        cv2.destroyAllWindows()
        if not render_overheads:
            self.get_logger().error('Display loop never ran; nothing to report.')
            return 1
        # What remains inside the reported number after this subtraction is the monitor's own
        # scanout, which UMI does not correct for either. ponytail: ~5-20 ms floor; a photodiode
        # would remove it and is not worth the rig.
        return self._report_camera(float(np.mean(render_overheads)))

    def _run_arm(self) -> int:
        """Sweep the commanded EEF pose along one axis and watch the TCP follow."""
        base = self._lookup_tcp()
        if base is None:
            return 1
        axis_idx = 'xyz'.index(self._axis)
        # Both the command and the measurement are expressed as displacement from where the arm
        # started, so the correlated pair is centred and the base position cancels.
        base_offset = getattr(base.transform.translation, self._axis)
        self.get_logger().warning(
            f'ARM WILL MOVE: sweeping {self._axis} by +/-{self._amplitude * 1e3:.0f} mm about the '
            f'current pose for {self._duration_s:g}s. Ctrl-C to abort.'
        )

        sampler = self.create_timer(1 / TF_SAMPLE_HZ, lambda: self._sample_tcp(axis_idx))
        commanded: list[tuple[float, float]] = []
        start = self._now()
        period = 1 / self._command_hz
        try:
            while (elapsed := self._now() - start) < self._duration_s:
                offset = self._amplitude * float(self._chirp(elapsed))
                pose = Pose()
                # Translate on one axis only and hold the orientation: a pure single-axis sweep is
                # what makes the recorded command a clean scalar to correlate against.
                pose.position.x = base.transform.translation.x
                pose.position.y = base.transform.translation.y
                pose.position.z = base.transform.translation.z
                setattr(pose.position, self._axis, base_offset + offset)
                pose.orientation = base.transform.rotation
                msg = PoseArray()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self._base_frame
                msg.poses = [pose]
                self._pub.publish(msg)
                commanded.append((self._now(), offset))
                time.sleep(period)
        finally:
            sampler.cancel()

        with self._lock:
            actual = [(t, x - base_offset) for t, x in self._actual]
        return self._report_xcorr(
            'latency.arm_exec', commanded, actual,
            note=(
                'This is plan+execute through MoveIt, not a transport delay: it is a distribution\n'
                '  dominated by planning time and it shifts with max_velocity_scaling. Measure it at\n'
                '  the scaling you will actually run at. It feeds _n_stale_actions, in units of\n'
                f'  action_dt={self._action_dt}s, so tens of ms of spread here is tolerable.'
            ),
        )

    def _publish_width(self, width: float) -> float:
        """Command one aperture, returning the instant it went out."""
        point = JointTrajectoryPoint()
        point.positions = [width]
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [GRIPPER_JOINT_NAME]
        msg.points = [point]
        self._pub.publish(msg)
        return self._now()

    def _wait_until_still(self) -> tuple[float, float] | None:
        """
        Block until the aperture stops changing, and report where it settled and how noisy it is.

        The resting noise is measured rather than assumed because it is what the onset threshold
        has to clear: a threshold under the encoder's own jitter would trigger on nothing at all
        and report a latency of zero.

        :returns: ``(resting width, peak-to-peak noise over the settle window)``, or None on timeout.
        """
        deadline = time.monotonic() + self._settle_timeout_s
        while time.monotonic() < deadline:
            time.sleep(GRIPPER_SETTLE_S)
            with self._lock:
                window = [w for t, w in self._actual if self._now() - t < GRIPPER_SETTLE_S]
            if len(window) < 3:
                continue
            if max(window) - min(window) < GRIPPER_STILL_TOL_M:
                return float(np.median(window)), float(max(window) - min(window))
        return None

    def _wait_for_onset(self, rest: float, t_command: float) -> float | None:
        """
        Return the stamp of the first sample that has moved off ``rest``, or None if none does.

        Keyed on the sample's own stamp rather than on when this loop noticed, so the polling rate
        here cannot inflate the answer.
        """
        deadline = time.monotonic() + self._settle_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                moved = [
                    t for t, w in self._actual
                    if t > t_command and abs(w - rest) > self._onset_threshold
                ]
            if moved:
                return min(moved)
            time.sleep(0.005)
        return None

    def _run_gripper(self) -> int:
        """
        Step the commanded aperture and time how long until the fingers respond.

        Deliberately **not** the chirp-and-cross-correlate that upstream UMI uses for the WSG and
        that the arm mode uses here. `fr3_gripper_bridge` quantises commands to
        ``min_command_period_s`` (0.25 s), supersedes each in-flight `Move` goal with the next, and
        drops anything inside its 5 mm deadband. Cross-correlation assumes the response is a
        delayed *linear echo* of the command, and a discrete goal-superseding commander is not one:
        driven at 0.6 Hz on hardware the hand fell most of a cycle behind, and the estimator
        faithfully reported the resulting phase lag — 1.2 s — as though it were a delay. The
        giveaway was that the answer grew with how much of the accelerating sweep it was given
        (0.41 s -> 0.94 -> 1.04 -> 1.20), where a real transport delay is invariant to that.

        A step response has no such assumption, and it is also what the bridge actually does in
        service. Note the ~0.25 s command quantisation is *inside* the number and belongs there:
        the policy's commands are quantised by the same timer.
        """
        low, high = self._center - self._step / 2, self._center + self._step / 2
        self.get_logger().warning(
            f'FINGERS WILL MOVE: {self._reps} steps between {low * 1e3:.0f} and {high * 1e3:.0f} mm. '
            'Nothing between the fingers. Ctrl-C to abort.'
        )
        lags: list[tuple[str, float]] = []
        noises = []
        for rep in range(self._reps):
            # Alternate direction so opening and closing are both sampled and the hand cannot
            # walk off to one end of its stroke.
            opening = rep % 2 == 0
            start, target = (low, high) if opening else (high, low)
            self._publish_width(start)
            settled = self._wait_until_still()
            if settled is None:
                self.get_logger().error(
                    f'Rep {rep + 1}: aperture never settled within {self._settle_timeout_s:g}s. '
                    'Is something between the fingers?'
                )
                return 1
            rest, noise = settled
            noises.append(noise)
            t_command = self._publish_width(target)
            onset = self._wait_for_onset(rest, t_command)
            if onset is None:
                self.get_logger().error(
                    f'Rep {rep + 1}: commanded {target * 1e3:.0f} mm and the hand never moved. Is '
                    'the bridge running with execute:=true? Without it every command is a no-op.'
                )
                return 1
            lags.append(('open' if opening else 'close', onset - t_command))
            self.get_logger().info(
                f'Rep {rep + 1}/{self._reps} ({"open" if opening else "close"}): '
                f'{(onset - t_command) * 1e3:.0f} ms'
            )
        return self._report_gripper_steps(lags, float(np.max(noises)))

    def _lookup_tcp(self):
        """Read the current TCP transform, or None (having logged why) if TF cannot supply it."""
        deadline = time.monotonic() + SUBSCRIBER_TIMEOUT_S
        last = ''
        while time.monotonic() < deadline:
            try:
                return self._tf_buffer.lookup_transform(
                    self._base_frame, self._eef_frame, rclpy.time.Time()
                )
            except Exception as e:  # noqa: BLE001 — tf2 raises several unrelated types
                last = str(e)
                time.sleep(0.2)
        self.get_logger().error(
            f'No {self._base_frame}->{self._eef_frame} transform after {SUBSCRIBER_TIMEOUT_S}s: {last}. '
            'Is fr3-bringup up on the NUC, and are the clocks in sync (see CLAUDE.md)?'
        )
        return None

    def _sample_tcp(self, axis_idx: int) -> None:
        """Record the TCP position on the swept axis, keyed by the transform's own stamp."""
        try:
            tf = self._tf_buffer.lookup_transform(self._base_frame, self._eef_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001 — a momentary gap is normal; the run tolerates it
            return
        stamp_s = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        value = (tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z)[axis_idx]
        with self._lock:
            # Sampling faster than the arm publishes, so most polls return the same transform
            # again. Duplicates would weight one instant heavily in the resampling.
            if self._actual and self._actual[-1][0] == stamp_s:
                return
            self._actual.append((stamp_s, value))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _report_gripper_steps(self, lags: list[tuple[str, float]], worst_noise_m: float) -> int:
        """Turn the per-rep step onsets into latency.gripper_exec, and judge the spread."""
        values = np.array([lag for _, lag in lags])
        opening = np.array([lag for d, lag in lags if d == 'open'])
        closing = np.array([lag for d, lag in lags if d == 'close'])
        median = float(np.median(values))
        with self._lock:
            samples = list(self._actual)

        lines = [
            '',
            '=' * 78,
            f'gripper command->motion: {median * 1e3:.0f} ms median '
            f'({values.min() * 1e3:.0f}-{values.max() * 1e3:.0f} ms over {len(values)} steps)',
            '=' * 78,
            f'  opening           {np.median(opening) * 1e3:.0f} ms median',
            f'  closing           {np.median(closing) * 1e3:.0f} ms median',
            f'  resting noise     {worst_noise_m * 1e3:.2f} mm worst, against a '
            f'{self._onset_threshold * 1e3:.1f} mm onset threshold',
            '',
        ]
        noisy = worst_noise_m >= self._onset_threshold
        if noisy:
            lines += [
                '  ** DO NOT PASTE THIS **',
                f'   - resting noise ({worst_noise_m * 1e3:.2f} mm) reaches the onset threshold '
                f'({self._onset_threshold * 1e3:.1f} mm), so the trigger may have fired on jitter',
                '     rather than on motion. Raise onset_threshold_m and re-run.',
                '',
            ]
        else:
            lines += [
                '  Paste into ros2_ws/src/polyumi_ros2/config/inference.yaml:',
                '',
                f'      gripper_exec: {median:.4f}',
                '',
                '  policy_client_node truncates the gripper chunk by this value alone, independently',
                '  of latency.arm_exec, so it goes in as measured — no arithmetic against the arm.',
                '',
                "  Most of the spread is the bridge's own min_command_period_s (0.25 s default): a",
                '  step published just after its timer fires waits nearly a full period. That',
                "  quantisation is real delay in service too — the policy's commands go through the",
                '  same timer — so it belongs in the number, and the median summarises it.',
                '',
            ]
        if len(samples) >= 3:
            interval = float(np.median(np.diff([t for t, _ in samples])))
            lines += [
                f'  Separately: /fr3_gripper/joint_states publishes every {interval * 1e3:.1f} ms, so',
                '  the honest bound on the OBSERVATION latency is half of that:',
                '',
                f'      latency.gripper: {interval / 2:.4f}',
                '',
                '  A different quantity from the step above, which is the ACTION side. Isolating the',
                '  observation half needs ground truth of the true aperture; franka_gripper exposes no',
                "  measure timestamp (UMI reads the WSG's), so this bound is as far as it goes.",
                '',
            ]
        print('\n'.join(lines))
        self._save_npz(
            method='step', latency_s=median, lags_s=values,
            directions=np.array([d for d, _ in lags]), worst_noise_m=worst_noise_m,
        )
        return 1 if noisy else 0

    def _report_xcorr(self, key, commanded, actual, note='') -> int:
        """Cross-correlate commanded against measured, judge the peak, and print the config line."""
        if len(commanded) < 2 or len(actual) < 2:
            self.get_logger().error(
                f'Not enough samples to correlate: {len(commanded)} commanded, {len(actual)} measured. '
                'Did the state topic publish at all?'
            )
            return 1
        t_target = np.array([t for t, _ in commanded])
        x_target = np.array([x for _, x in commanded])
        t_actual = np.array([t for t, _ in actual])
        x_actual = np.array([x for _, x in actual])
        try:
            # force_positive: a command cannot precede its own response, so a negative peak is
            # always an artifact rather than a measurement.
            latency, info = get_latency(x_target, t_target, x_actual, t_actual, force_positive=True)
        except ValueError as e:
            self.get_logger().error(f'Estimation failed: {e}')
            return 1

        travelled = float(np.ptp(x_actual))
        lines = [
            '',
            '=' * 78,
            f'{self._mode} latency: {latency * 1e3:.1f} ms',
            '=' * 78,
            f'  peak correlation  {info["peak_corr"]:.3f}   (>{MIN_PEAK_CORR} to be a match at all)',
            f'  peak width        {info["peak_width_s"] * 1e3:.0f} ms  '
            f'(<{MAX_PEAK_WIDTH_S * 1e3:.0f} ms to localise the lag)',
            f'  samples           {len(commanded)} commanded, {len(actual)} measured',
            f'  measured travel   {travelled * 1e3:.1f} mm peak-to-peak',
            '',
        ]
        bad = []
        if info['pinned']:
            bad.append(
                f'the winning lag sits exactly on the {info["max_lag_s"]}s search bound, so this is '
                'the clamp and not a peak. Either the plant lags more than the bound (raise it), or '
                'it is not tracking the sweep at all and the correlation has nowhere to peak.'
            )
        if info['peak_corr'] < MIN_PEAK_CORR:
            bad.append(
                'peak correlation is too low — the measured signal does not track the command. '
                'Check the execute flag is set and the plant actually moved.'
            )
        if info['peak_width_s'] > MAX_PEAK_WIDTH_S:
            bad.append(
                'peak is too broad to localise — raise chirp_f1_hz or amplitude_m, or lengthen '
                'duration_s, and re-run rather than believing this number.'
            )
        if bad:
            lines += ['  ** DO NOT PASTE THIS **'] + [f'   - {b}' for b in bad] + ['']
        else:
            lines += [
                '  Paste into ros2_ws/src/polyumi_ros2/config/inference.yaml:',
                '',
                f'      {key}: {latency:.4f}',
                '',
            ]
        if note:
            lines += [f'  {note}', '']
        print('\n'.join(lines))

        self._save_npz(
            method='xcorr', latency_s=latency, t_target=t_target, x_target=x_target,
            t_actual=t_actual, x_actual=x_actual,
            peak_corr=info['peak_corr'], peak_width_s=info['peak_width_s'],
        )
        if self._plot:
            self._plot_xcorr(latency, info)
        return 1 if bad else 0

    def _report_camera(self, render_overhead_s: float) -> int:
        """Decode the buffered frames and report photon->stamp, plus the stamp->arrival diagnostic."""
        with self._lock:
            frames = list(self._frames)
        if not frames:
            self.get_logger().error('No camera frames arrived. Is the v4l2 node publishing?')
            return 1

        detector = cv2.QRCodeDetector()
        decoded = []
        arrival_deltas = []
        for stamp_s, arrival_s, gray in frames:
            arrival_deltas.append(arrival_s - stamp_s)
            decoded.append((stamp_s, decode_qr(detector, gray)))
        first_stamp = first_stamp_per_code(decoded)

        if len(first_stamp) < 3:
            self.get_logger().error(
                f'Only {len(first_stamp)} QR codes decoded from {len(frames)} frames. Fill the frame '
                'with the screen, check focus, and avoid glare — then re-run.'
            )
            return 1

        offsets = np.array([stamp - qr for qr, stamp in first_stamp.items()]) - render_overhead_s
        latency = float(np.mean(offsets))
        arrival = np.array(arrival_deltas)
        print('\n'.join([
            '',
            '=' * 78,
            f'latency.gopro: {latency * 1e3:.1f} ms  (std {np.std(offsets) * 1e3:.1f} ms, '
            f'n={len(offsets)})',
            '=' * 78,
            f'  screen render overhead subtracted: {render_overhead_s * 1e3:.1f} ms',
            '',
            '  Paste into ros2_ws/src/polyumi_ros2/config/inference.yaml:',
            '',
            f'      gopro: {latency:.4f}',
            '',
            '  UMI measures 0.125-0.17 s for the same GoPro->HDMI->Elgato chain. Far outside that',
            '  means the rig, not the pipeline. Monitor scanout is still inside this number.',
            '',
            f'  Separately, stamp->arrival was {np.median(arrival) * 1e3:.0f} ms median / '
            f'{np.max(arrival) * 1e3:.0f} ms max.',
            '  That is the YUYV convert and transport, which happen AFTER v4l2 stamps the frame, so',
            '  it is NOT part of latency.gopro. It is what max_image_age_s has to tolerate — keep',
            '  that above the max above.',
            '',
        ]))
        self._save_npz(
            method='qr', latency_s=latency,
            qr_times=np.array(list(first_stamp.keys())),
            recv_stamps=np.array(list(first_stamp.values())),
            arrival_deltas=arrival, render_overhead_s=render_overhead_s,
        )
        return 0

    def _save_npz(self, **arrays) -> None:
        """Dump the raw series so a marginal run can be re-judged without re-booking hardware."""
        path = self._output_npz or f'latency_{self._mode}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.npz'
        np.savez(path, mode=self._mode, **arrays)
        self.get_logger().info(f'Raw series saved to {path}')

    def _plot_xcorr(self, latency: float, info: dict) -> None:
        """Show UMI's three-panel view: the correlation curve, the raw pair, and the aligned pair."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        _, axes = plt.subplots(3, 1, figsize=(10, 8))
        axes[0].plot(info['lags'], info['correlation'])
        axes[0].axvline(latency, color='r', ls='--')
        axes[0].set_title(f'correlation (peak at {latency * 1e3:.1f} ms)')
        axes[0].set_xlabel('lag (s)')
        axes[0].set_xlim(-0.5, 1.0)
        axes[1].plot(info['t_samples'], info['x_target'], label='commanded')
        axes[1].plot(info['t_samples'], info['x_actual'], label='measured')
        axes[1].set_title('as recorded')
        axes[1].legend()
        axes[2].plot(info['t_samples'], info['x_target'], label='commanded')
        axes[2].plot(info['t_samples'] - latency, info['x_actual'], label='measured, shifted back')
        axes[2].set_title(f'aligned with latency={latency * 1e3:.1f} ms — these should overlay')
        axes[2].legend()
        plt.tight_layout()
        plt.show()


def main():
    """Spin the node on a background thread and run the probe in the foreground."""
    rclpy.init()
    try:
        node = LatencyProbe()
    except ValueError as e:
        print(f'latency_probe: {e}')
        rclpy.shutdown()
        return 2
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        code = node.run()
    except KeyboardInterrupt:
        code = 130
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == '__main__':
    raise SystemExit(main())

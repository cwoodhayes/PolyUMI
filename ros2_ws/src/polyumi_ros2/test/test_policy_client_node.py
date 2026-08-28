"""
Tests for policy_client_node's latency math and configuration validation.

These cover the time arithmetic that decides *which* pose gets paired with an image and *how
much* of an action chunk is still valid. Both are easy to get subtly wrong in a way no runtime
error reveals — a sign flip in the pose lookup just trains/deploys on a slightly-off pose, and
a bad stale count just moves the arm to the wrong waypoint — so they are pinned here rather
than left to hardware testing to notice.
"""

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.clock import ClockType
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState

from polyumi_inference import ActionChunk, Observation, TransportError
from polyumi_ros2.policy_client_node import PolicyClientNode

#: Stand-in for the tests that drive _post_and_act directly. Its contents never reach a server --
#: those tests mock the client -- but it has to be the type the method now takes.
_OBS = Observation(
    channels={'camera0_rgb': np.zeros((2, 4, 4, 3), dtype=np.uint8), 'agent_pos': np.zeros((2, 8))},
    n_obs_steps=2,
    n_action_steps=8,
)

BASE_FRAME = 'fr3_link0'
EEF_FRAME = 'polyumi_tcp'


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    """Init/shutdown rclpy once for the whole module."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def make_node():
    """Construct PolicyClientNode with parameter overrides, destroying it afterwards."""
    nodes = []

    def _make(**params):
        overrides = [Parameter(name, value=value) for name, value in params.items()]
        node = PolicyClientNode(parameter_overrides=overrides)
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node.destroy_node()


def _t(seconds: float) -> Time:
    """Build a ROS_TIME instant, matching the clock type the node's own clock produces."""
    return Time(nanoseconds=int(seconds * 1e9), clock_type=ClockType.ROS_TIME)


class _FakeClock:
    """Stands in for the node's clock so the tests do not race real time."""

    def __init__(self, now: Time):
        self._now = now

    def now(self) -> Time:
        return self._now


# ----------------------------------------------------------------------
# Configuration validation
# ----------------------------------------------------------------------


def test_valid_config_constructs(make_node):
    """The values actually shipped in config/inference.yaml must pass validation."""
    node = make_node(
        control_hz=10.0,
        **{'latency.gopro': 0.05, 'latency.proprio': 0.01, 'latency.arm_exec': 0.01, 'buffers.ee_pose_s': 1.0},
    )
    assert node._action_dt == pytest.approx(0.1)


@pytest.mark.parametrize(
    'params, expected',
    [
        ({'control_hz': 0.0}, 'control_hz must be > 0'),
        ({'control_hz': -5.0}, 'control_hz must be > 0'),
        ({'buffers.ee_pose_s': 0.0}, 'buffers.ee_pose_s must be > 0'),
        ({'n_obs_steps': 0}, 'n_obs_steps must be >= 1'),
        ({'n_action_steps': 0}, 'n_action_steps must be >= 1'),
        ({'steps_per_inference': 0}, 'steps_per_inference must be >= 1'),
        ({'latency.gopro': -0.1}, 'latency.gopro must be >= 0'),
        ({'latency.arm_exec': -0.1}, 'latency.arm_exec must be >= 0'),
    ],
)
def test_invalid_config_rejected(make_node, params, expected):
    """Each bad value fails at construction with a message naming the offending parameter."""
    with pytest.raises(ValueError, match=expected):
        make_node(**params)


def test_config_errors_are_reported_together(make_node):
    """All violations surface at once, so fixing config is not one-restart-per-typo."""
    with pytest.raises(ValueError) as excinfo:
        make_node(control_hz=0.0, **{'latency.proprio': -0.5})
    assert 'control_hz must be > 0' in str(excinfo.value)
    assert 'latency.proprio must be >= 0' in str(excinfo.value)


def test_tf_buffer_must_cover_the_compensated_lookup(make_node):
    """
    A TF buffer shorter than the lookup offset is rejected rather than left to fail at runtime.

    Without this check the node starts happily and then every _lookup_agent_pos call asks for a
    transform the buffer has already evicted, producing endless 'TF lookup failed' warnings
    that point nowhere near the actual cause.
    """
    with pytest.raises(ValueError, match='compensated'):
        make_node(**{'latency.gopro': 0.5, 'latency.proprio': 0.0, 'buffers.ee_pose_s': 0.2})


# ----------------------------------------------------------------------
# Stale-action truncation
# ----------------------------------------------------------------------


def _n_stale_at(node, *, obs_age_s: float, latency_act: float | None = None) -> int:
    """Evaluate _n_stale_actions for an observation captured obs_age_s ago (may be negative)."""
    now = _t(100.0)
    t_obs = _t(100.0 - obs_age_s)
    if latency_act is None:
        latency_act = node._latency_act
    with patch.object(node, 'get_clock', return_value=_FakeClock(now)):
        return node._n_stale_actions(t_obs, latency_act)


def test_n_stale_actions_typical(make_node):
    """A 130 ms-old observation plus 10 ms of arm latency invalidates the first 2 of 8 actions."""
    node = make_node(control_hz=10.0, **{'latency.arm_exec': 0.01})
    # 0.13 + 0.01 = 0.14s of latency, spanning 1.4 action slots -> the first 2 have elapsed.
    assert _n_stale_at(node, obs_age_s=0.13) == 2


def test_n_stale_actions_fresh_observation(make_node):
    """With no latency at all, nothing is stale and the whole chunk survives."""
    node = make_node(control_hz=10.0, **{'latency.arm_exec': 0.0})
    assert _n_stale_at(node, obs_age_s=0.0) == 0


def test_n_stale_actions_clamps_at_zero(make_node):
    """
    Regression: a future t_obs must not produce a negative count.

    If the camera driver's stamps run ahead of this node's clock, elapsed time goes negative.
    An unclamped count of -1 makes the caller's `actions[n_stale:]` keep only the *last*
    action — silently jumping the arm to the far-future tail of the trajectory instead of
    dropping nothing. The empty-chunk guard cannot catch it, because that slice is non-empty.
    """
    node = make_node(control_hz=10.0, **{'latency.arm_exec': 0.01})
    n_stale = _n_stale_at(node, obs_age_s=-0.2)  # stamped 200 ms in the future

    assert n_stale == 0
    assert list(range(8))[n_stale:] == list(range(8))  # whole chunk kept, nothing reordered


def test_n_stale_actions_whole_chunk_stale(make_node):
    """An observation older than the chunk span invalidates every action in it."""
    node = make_node(control_hz=10.0, **{'latency.arm_exec': 0.01})
    n_stale = _n_stale_at(node, obs_age_s=1.0)  # 1.0s vs an 8-slot, 0.8s chunk

    assert n_stale >= 8
    assert list(range(8))[n_stale:] == []  # caller's empty-slice guard fires


def test_arm_and_gripper_are_truncated_by_their_own_latencies(make_node):
    """
    The whole point of the split: a faster device keeps actions the slower one has to drop.

    Both bridges used to share one slice cut with the ARM's latency, so the hand inherited the
    arm's lead and acted that much too early — the old gripper bridge's gripper_lead_steps existed
    only to index back out of it, and could add lead but never remove it. Measured on hardware the
    hand beat the arm by 188 ms, i.e. about two action steps.
    """
    node = make_node(control_hz=10.0, **{'latency.arm_exec': 0.702, 'latency.gripper_exec': 0.514})
    n_arm = _n_stale_at(node, obs_age_s=0.1, latency_act=node._latency_act)
    n_grip = _n_stale_at(node, obs_age_s=0.1, latency_act=node._latency_act_gripper)

    assert n_arm == 9  # (0.100 + 0.702) / 0.1, rounded up
    assert n_grip == 7  # (0.100 + 0.514) / 0.1, rounded up
    # The gripper keeps two more waypoints, which is exactly the 188 ms it is faster.
    assert n_arm - n_grip == 2


def test_a_chunk_too_stale_for_the_arm_can_still_drive_the_gripper(make_node):
    """
    The faster device must not be stalled by the slower one running out of chunk.

    With the shared slice, an empty arm chunk returned early and neither device was commanded.
    Since the arm is currently 702 ms against an 8-step (0.8 s) chunk, that is not hypothetical.
    """
    node = make_node(control_hz=10.0, **{'latency.arm_exec': 0.702, 'latency.gripper_exec': 0.514})
    chunk = list(range(8))
    # 0.150 + 0.702 = 0.852s spans past the whole 0.8s chunk; 0.150 + 0.514 = 0.664s does not.
    n_arm = _n_stale_at(node, obs_age_s=0.15, latency_act=node._latency_act)
    n_grip = _n_stale_at(node, obs_age_s=0.15, latency_act=node._latency_act_gripper)

    assert chunk[n_arm:] == []  # arm has nothing left
    assert chunk[n_grip:] == [7]  # gripper still has a waypoint to act on


# ----------------------------------------------------------------------
# Time-aligned pose lookup
# ----------------------------------------------------------------------


def _push_ramp_tf(node, *, t0: float = 0.6, t1: float = 1.4, step: float = 0.1) -> None:
    """
    Fill the node's TF buffer with a ramp where x-position equals the timestamp.

    Encoding time into position is what lets a test assert *which instant* was looked up: the
    returned x is a direct readout of the effective query time.

    The default span is deliberately shorter than buffers.ee_pose_s: tf2 evicts relative to the
    *newest* transform it holds, so a ramp wider than the cache silently drops its own early
    samples and the lookup fails as extrapolation-into-the-past. That eviction rule is exactly
    why _validate_params requires the buffer to outlast the compensated lookup offset.
    """
    t = t0
    while t <= t1 + 1e-9:
        tf = TransformStamped()
        tf.header.stamp = _t(t).to_msg()
        tf.header.frame_id = BASE_FRAME
        tf.child_frame_id = EEF_FRAME
        tf.transform.translation.x = t
        tf.transform.rotation.w = 1.0
        node._tf_buffer.set_transform(tf, 'test_authority')
        t += step


def test_lookup_agent_pos_targets_the_frames_capture_instant(make_node):
    """
    The pose is looked up at the image's *capture* instant, not its arrival stamp.

    latency.gopro is how long the frame spent in the capture pipeline (GoPro encode -> HDMI ->
    capture card -> v4l2 dequeue) before being stamped, so the pose that belongs with it is the
    one from gopro seconds earlier. Pairing the image with the pose from its *stamp* instead
    would hand the policy an observation whose two halves disagree by that much.
    """
    node = make_node(**{'latency.gopro': 0.05, 'latency.proprio': 0.0, 'buffers.ee_pose_s': 1.0})
    _push_ramp_tf(node)

    agent_pos = node._lookup_agent_pos(image_stamp=_t(1.0))

    assert agent_pos is not None
    # x == query time, so 1.0 - 0.05 = 0.95 proves the offset was applied in the right direction.
    assert agent_pos[0] == pytest.approx(0.95, abs=1e-6)


def test_lookup_agent_pos_adds_proprio_latency(make_node):
    """
    Proprio latency moves the query *forward*, which looks wrong until you follow the stamps.

    TF entries are stamped when the measurement is published, so an entry stamped T describes
    where the arm was at T - proprio. To read the arm's true state at capture instant t_cap,
    the entry to ask for is the one stamped t_cap + proprio. This mirrors UMI, which corrects
    each stream to t_recv - latency and interpolates the low-dim streams onto the camera's
    corrected clock.
    """
    node = make_node(**{'latency.gopro': 0.05, 'latency.proprio': 0.01, 'buffers.ee_pose_s': 1.0})
    _push_ramp_tf(node)

    agent_pos = node._lookup_agent_pos(image_stamp=_t(1.0))

    assert agent_pos is not None
    # 1.0 - 0.05 + 0.01 = 0.96, i.e. proprio pulls the query back toward the present.
    assert agent_pos[0] == pytest.approx(0.96, abs=1e-6)


def test_lookup_agent_pos_interpolates_between_samples(make_node):
    """
    A query landing between two TF samples interpolates rather than snapping to the nearest.

    This is why the node relies on tf2's buffer instead of a hand-rolled pose ring buffer: the
    compensated instant almost never coincides with a sample.
    """
    node = make_node(**{'latency.gopro': 0.05, 'latency.proprio': 0.0, 'buffers.ee_pose_s': 1.0})
    _push_ramp_tf(node, step=0.1)  # samples at 0.9 and 1.0, none at 0.95

    agent_pos = node._lookup_agent_pos(image_stamp=_t(1.0))

    assert agent_pos is not None
    assert agent_pos[0] == pytest.approx(0.95, abs=1e-6)  # midpoint, not 0.9 or 1.0


def test_lookup_agent_pos_returns_none_when_tf_is_empty(make_node):
    """With no TF data the lookup reports failure instead of raising into the control loop."""
    node = make_node(**{'latency.gopro': 0.05, 'buffers.ee_pose_s': 1.0})

    assert node._lookup_agent_pos(image_stamp=_t(1.0)) is None


def test_lookup_agent_pos_shape_and_orientation(make_node):
    """The returned vector is the 8-wide [xyz, quat, gripper] the server contract expects."""
    node = make_node(**{'latency.gopro': 0.0, 'latency.proprio': 0.0, 'buffers.ee_pose_s': 1.0})
    _push_ramp_tf(node)

    agent_pos = node._lookup_agent_pos(image_stamp=_t(1.0))

    assert agent_pos is not None
    assert agent_pos.shape == (8,)
    assert agent_pos.dtype == np.float64
    np.testing.assert_allclose(agent_pos[3:7], [0.0, 0.0, 0.0, 1.0], atol=1e-9)


# ----------------------------------------------------------------------
# Episode /reset and viz-only preview (arm dry-run wiring)
# ----------------------------------------------------------------------


def test_reset_url_derives_from_predict_url(make_node):
    """The /reset URL is derived from the predict URL's base, with or without a trailing slash."""
    node = make_node(inference_server_url='http://sheep:8000/predict_cartesian/')
    assert node._reset_url == 'http://sheep:8000/reset'

    node2 = make_node(inference_server_url='http://sheep:8000/predict_cartesian')
    assert node2._reset_url == 'http://sheep:8000/reset'


def test_reset_episode_sends_start_pose_once(make_node):
    """
    _reset_episode hands the given pose to the client and latches _episode_reset_done.

    The URL and the JSON shaping belong to PolicyClient and are tested there; what this node owns
    is sending the pose exactly once per episode.
    """
    node = make_node(inference_server_url='http://sheep:8000/predict_cartesian/')
    agent_pos = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0])

    with patch.object(node._client, 'reset', return_value={'status': 'ok'}) as post:
        node._reset_episode(agent_pos)

    assert node._episode_reset_done is True
    post.assert_called_once()
    assert np.array_equal(post.call_args[0][0], agent_pos)


def test_reset_episode_not_latched_on_failure(make_node):
    """A failed /reset leaves the flag unset so the next tick retries."""
    node = make_node()
    with patch.object(node._client, 'reset', side_effect=TransportError('nope', url='u')):
        node._reset_episode(np.zeros(8))
    assert node._episode_reset_done is False


def test_preview_published_full_chunk_without_execution(make_node):
    """
    The preview publishes the FULL commanded chunk even with execute_motion off and all-stale.

    This is the eyeball for the arm dry-run: the whole policy output must reach Foxglove
    regardless of execution or staleness, while nothing is published to the execution topic.
    """
    node = make_node(publish_preview=True)  # execute_motion defaults to False
    assert node._target_pub is None  # nothing can drive the arm
    assert node._preview_pub is not None

    actions = [[float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] for i in range(8)]
    now = _t(100.0)
    t_obs = _t(98.0)  # 2s old -> every action is stale, so the chunk is dropped for execution
    with (
        patch.object(node._client, 'predict', return_value=ActionChunk(actions)),
        patch.object(node, 'get_clock', return_value=_FakeClock(now)),
        patch.object(node._preview_pub, 'publish') as preview_pub,
    ):
        node._post_and_act(obs=_OBS, t_obs=t_obs)

    preview_pub.assert_called_once()
    msg = preview_pub.call_args[0][0]
    assert len(msg.poses) == 8  # full chunk, not the post-stale-drop subset
    assert msg.header.frame_id == node._base_frame


def test_preview_disabled_creates_no_publisher(make_node):
    """publish_preview=false suppresses the preview publisher entirely."""
    node = make_node(publish_preview=False)
    assert node._preview_pub is None


def test_max_image_age_auto_default(make_node):
    """max_image_age_s=0 resolves to the auto value: half a control period at 10 Hz."""
    node = make_node(control_hz=10.0)
    assert node._max_image_age_s == pytest.approx(0.05)  # max(2/60, 0.5/10) = 0.05


def test_max_image_age_override(make_node):
    """A positive max_image_age_s wins over the auto formula (for slow camera paths)."""
    node = make_node(control_hz=10.0, max_image_age_s=0.3)
    assert node._max_image_age_s == pytest.approx(0.3)


# ----------------------------------------------------------------------
# Receding-horizon stride (inference cadence)
# ----------------------------------------------------------------------


def _drive_ticks(node, n_ticks: int) -> int:
    """
    Run n_ticks control ticks with a full obs buffer, returning how many ran inference.

    Bypasses the camera/TF plumbing (image cached, pose mocked, buffer pre-filled, reset
    latched) so the test isolates the stride gate alone: the only thing that varies tick to
    tick is _inference_phase, so the count of _submit_inference calls is the inference cadence.

    _submit_inference is the seam, not _post_and_act: the tick's whole job is to decide and hand
    off, and the request itself runs on another thread. Counting there would be counting the
    worker's schedule as well as the stride.
    """
    now = _t(100.0)
    node._latest_image = np.zeros((4, 4, 3), dtype=np.uint8)
    node._latest_image_stamp = now
    # Pre-fill the obs buffer so every tick is a full-buffer tick (skips the fill-up ramp),
    # and latch the reset so the episode-start POST doesn't interfere.
    fixed_pose = np.zeros(8)
    for _ in range(node._n_obs_steps):
        node._obs_buffer.append((node._latest_image, fixed_pose))
    node._episode_reset_done = True

    infer_calls = []
    with (
        patch.object(node, 'get_clock', return_value=_FakeClock(now)),
        patch.object(node, '_lookup_agent_pos', return_value=fixed_pose),
        patch.object(node, '_submit_inference', side_effect=lambda *a, **k: infer_calls.append(1)),
    ):
        for _ in range(n_ticks):
            node._control_tick()
    return len(infer_calls)


def test_stride_runs_inference_every_n_ticks(make_node):
    """With steps_per_inference=3, inference fires on ticks 1, 4, 7, 10 — not every tick."""
    node = make_node(control_hz=10.0, n_obs_steps=2, steps_per_inference=3)
    assert _drive_ticks(node, 10) == 4


def test_stride_one_infers_every_tick(make_node):
    """steps_per_inference=1 is the old behaviour: an inference on every tick."""
    node = make_node(control_hz=10.0, n_obs_steps=2, steps_per_inference=1)
    assert _drive_ticks(node, 10) == 10


def test_stride_first_full_tick_infers_immediately(make_node):
    """The very first full-buffer tick infers (phase starts at 0), not after a stride delay."""
    node = make_node(control_hz=10.0, n_obs_steps=2, steps_per_inference=6)
    assert _drive_ticks(node, 1) == 1


def test_tf_use_latest_ignores_stamp(make_node):
    """tf_use_latest looks up the newest transform regardless of the requested image stamp."""
    node = make_node(tf_use_latest=True, **{'latency.gopro': 0.05, 'buffers.ee_pose_s': 1.0})
    _push_ramp_tf(node)  # ramp x==timestamp over [0.6, 1.4]

    # A stamp well before the ramp would extrapolate-into-past in time-aligned mode; with
    # tf_use_latest it returns the newest sample (x == 1.4) instead.
    agent_pos = node._lookup_agent_pos(image_stamp=_t(0.1))
    assert agent_pos is not None
    assert agent_pos[0] == pytest.approx(1.4, abs=1e-6)


# ----------------------------------------------------------------------
# Gripper — observation
# ----------------------------------------------------------------------


def _push_gripper(node, stamp_s: float, width_m: float) -> None:
    """Feed one gripper joint state, split across the two fingers as the FR3 reports it."""
    msg = JointState()
    msg.header.stamp = _t(stamp_s).to_msg()
    msg.name = ['fr3_finger_joint1', 'fr3_finger_joint2']
    msg.position = [width_m / 2.0, width_m / 2.0]
    node._gripper_cb(msg)


def test_gripper_width_sums_the_two_finger_joints(make_node):
    """
    Aperture is the SUM of the finger positions, each of which is half the opening.

    Reading position[0] alone (or the wrong joint) would halve every width the policy sees,
    which is exactly the kind of error that looks plausible in a log.
    """
    node = make_node(**{'latency.gopro': 0.0, 'latency.gripper': 0.0})
    _push_gripper(node, 1.0, 0.06)

    assert node._gripper_width_at(_t(1.0)) == pytest.approx(0.06)


def test_gripper_width_lands_in_agent_pos_index_7(make_node):
    """The measured aperture reaches agent_pos[7], converted into policy units."""
    node = make_node(
        gripper_min_width_m=0.005, **{'latency.gopro': 0.0, 'latency.proprio': 0.0, 'latency.gripper': 0.0}
    )
    _push_ramp_tf(node)
    _push_gripper(node, 1.0, 0.06)

    agent_pos = node._lookup_agent_pos(image_stamp=_t(1.0))

    assert agent_pos is not None
    assert agent_pos[7] == pytest.approx(0.055)  # 0.06 aperture - 0.005 closed aperture


def test_gripper_width_interpolates_to_the_capture_instant(make_node):
    """
    The width is sampled at the frame's capture instant, like the pose — not at tick time.

    Encoding time into width (width == timestamp) makes the returned value a direct readout of
    which instant was queried, the same trick _push_ramp_tf uses for the pose.
    """
    node = make_node(**{'latency.gopro': 0.05, 'latency.gripper': 0.0})
    for t in (0.8, 0.9, 1.0):
        _push_gripper(node, t, t)

    # 1.0 - 0.05 = 0.95, i.e. between the 0.9 and 1.0 samples rather than snapped to either.
    assert node._gripper_width_at(_t(0.95)) == pytest.approx(0.95, abs=1e-6)
    assert node._gripper_width_policy_units(image_stamp=_t(1.0)) == pytest.approx(0.95, abs=1e-6)


def test_gripper_latency_shifts_the_query_forward(make_node):
    """latency.gripper moves the query the same direction proprio does for the pose."""
    node = make_node(**{'latency.gopro': 0.05, 'latency.gripper': 0.01})
    for t in (0.8, 0.9, 1.0):
        _push_gripper(node, t, t)

    # 1.0 - 0.05 + 0.01 = 0.96
    assert node._gripper_width_policy_units(image_stamp=_t(1.0)) == pytest.approx(0.96, abs=1e-6)


def test_gripper_width_holds_endpoints_rather_than_extrapolating(make_node):
    """Outside the cached span the nearest sample is held; extrapolation would invent motion."""
    node = make_node()
    _push_gripper(node, 1.0, 0.04)
    _push_gripper(node, 1.1, 0.05)

    assert node._gripper_width_at(_t(0.5)) == pytest.approx(0.04)
    assert node._gripper_width_at(_t(5.0)) == pytest.approx(0.05)


def test_missing_gripper_state_falls_back_to_closed(make_node):
    """
    With no gripper state the tick still runs, substituting the closed width.

    This is what keeps arm-only bringup (motion_only, no hand attached) working rather than
    stalling the whole control loop on a device that isn't there.
    """
    node = make_node(gripper_min_width_m=0.005, **{'latency.gopro': 0.0, 'latency.proprio': 0.0})
    _push_ramp_tf(node)

    agent_pos = node._lookup_agent_pos(image_stamp=_t(1.0))

    assert agent_pos is not None
    assert agent_pos[7] == pytest.approx(0.0)  # fully closed is 0.0 in policy units, by definition


def test_require_gripper_state_skips_the_tick_instead(make_node):
    """With require_gripper_state, a missing width drops the tick like a failed TF lookup."""
    node = make_node(require_gripper_state=True, **{'latency.gopro': 0.0, 'latency.proprio': 0.0})
    _push_ramp_tf(node)  # pose is available; only the gripper is missing

    assert node._lookup_agent_pos(image_stamp=_t(1.0)) is None


def test_malformed_gripper_state_is_ignored(make_node):
    """A joint state with too few positions is dropped rather than crashing the callback."""
    node = make_node()
    msg = JointState()
    msg.header.stamp = _t(1.0).to_msg()
    msg.name = ['fr3_finger_joint1']
    msg.position = [0.02]
    node._gripper_cb(msg)

    assert node._gripper_width_at(_t(1.0)) is None


def test_tf_use_latest_takes_the_newest_gripper_sample(make_node):
    """
    Under tf_use_latest the gripper reads the NEWEST sample, not the oldest.

    tf2 spells "latest available" as a zero ``Time()``, but this buffer is hand-rolled and a zero
    stamp is simply an instant before every sample — passing the tf2 sentinel straight through
    returns ``samples[0]``, i.e. the oldest entry, which at the buffer's depth is seconds stale.
    A dry run is exactly where someone would be watching whether the width tracks, so it has to
    be the fresh end.
    """
    node = make_node(tf_use_latest=True)
    for stamp, width in ((1.0, 0.01), (1.1, 0.05), (1.2, 0.08)):
        _push_gripper(node, stamp, width)

    assert node._gripper_width_at(None) == pytest.approx(0.08)
    assert node._gripper_width_policy_units(image_stamp=_t(1.2)) == pytest.approx(0.08)


def test_stale_gripper_state_holds_the_last_width_and_warns(make_node):
    """
    A gripper topic that *stops* is caught by age, and the last width is held rather than zeroed.

    _gripper_width_at cannot see this failure — holding its newest endpoint looks identical to a
    slow publisher — so without the age check the policy is fed a frozen width indefinitely and
    in silence. Holding beats substituting closed: if the hand stopped reporting mid-grasp,
    "closed" is the bigger lie.
    """
    node = make_node(max_gripper_age_s=0.5, **{'latency.gopro': 0.0})
    _push_gripper(node, 1.0, 0.06)

    with (
        patch.object(node, 'get_clock', return_value=_FakeClock(_t(3.0))),
        patch.object(node, '_warn_throttled') as warn,
    ):
        width = node._gripper_width_policy_units(image_stamp=_t(3.0))

    assert width == pytest.approx(0.06)
    assert 'gripper topic died' in warn.call_args[0][0]


def test_stale_gripper_state_skips_the_tick_when_required(make_node):
    """With require_gripper_state, a stale width drops the tick like a missing one."""
    node = make_node(require_gripper_state=True, max_gripper_age_s=0.5, **{'latency.gopro': 0.0})
    _push_gripper(node, 1.0, 0.06)

    with patch.object(node, 'get_clock', return_value=_FakeClock(_t(3.0))):
        assert node._gripper_width_policy_units(image_stamp=_t(3.0)) is None


def test_fresh_gripper_state_is_not_treated_as_stale(make_node):
    """A width inside the age limit passes even under require_gripper_state."""
    node = make_node(
        require_gripper_state=True,
        max_gripper_age_s=0.5,
        **{'latency.gopro': 0.0},
    )
    _push_gripper(node, 1.0, 0.06)

    with patch.object(node, 'get_clock', return_value=_FakeClock(_t(1.2))):
        assert node._gripper_width_policy_units(image_stamp=_t(1.0)) == pytest.approx(0.06)


def test_gripper_age_check_can_be_disabled(make_node):
    """max_gripper_age_s <= 0 turns the check off, for setups with an odd clock or rate."""
    node = make_node(
        require_gripper_state=True,
        max_gripper_age_s=0.0,
        **{'latency.gopro': 0.0},
    )
    _push_gripper(node, 1.0, 0.06)

    with patch.object(node, 'get_clock', return_value=_FakeClock(_t(100.0))):
        assert node._gripper_width_policy_units(image_stamp=_t(100.0)) == pytest.approx(0.06)


# ----------------------------------------------------------------------
# Gripper — command chunk
# ----------------------------------------------------------------------


def _actions_with_grip(widths):
    """Build 8-vector actions whose only interesting component is the gripper width."""
    return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, w] for w in widths]


def test_gripper_trajectory_converts_to_robot_units(make_node):
    """
    Widths are converted to jaw aperture here, so the NUC bridge needs no calibration.

    Keeping the conversion on this side means one place holds the calibration, and the bridge can
    stay a dumb executor of the metres it is handed.
    """
    node = make_node(control_hz=10.0, gripper_min_width_m=0.005, gripper_max_width_m=0.08)

    msg = node._actions_to_gripper_trajectory(_actions_with_grip([0.005, 0.025, 0.5]), _t(100.0))

    assert [p.positions[0] for p in msg.points] == pytest.approx([0.01, 0.03, 0.08])


def test_gripper_trajectory_carries_chunk_timing(make_node):
    """
    Each point names an ABSOLUTE instant: header.stamp + time_from_start, anchored at t_obs.

    franka_hand_node holds these as a horizon and asks which are still reachable, so the times
    have to be a schedule rather than a shape. The anchor leads t_obs by latency.gripper_exec, so
    each width is commanded that far ahead of when the fingers should be there.
    """
    node = make_node(control_hz=10.0, **{'latency.gripper_exec': 0.38})

    msg = node._actions_to_gripper_trajectory(_actions_with_grip([0.02] * 3), _t(100.0))

    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    assert stamp == pytest.approx(100.0 - 0.38)
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points]
    assert times == pytest.approx([0.0, 0.1, 0.2])
    assert msg.joint_names == ['fr3_gripper_width']


def test_gripper_trajectory_numbers_from_the_preslice_index(make_node):
    """
    A stale-drop must not slide the surviving waypoints earlier.

    Numbering the survivors from zero would give back exactly the lead the drop removed, so the
    hand would be commanded to reach each width at the instant the DROPPED one was due.
    """
    node = make_node(control_hz=10.0, **{'latency.gripper_exec': 0.0})

    msg = node._actions_to_gripper_trajectory(_actions_with_grip([0.02] * 2), _t(100.0), first_index=3)

    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points]
    assert times == pytest.approx([0.3, 0.4])


def test_gripper_and_pose_chunks_stay_index_aligned(make_node):
    """
    Both execution topics get the same number of waypoints, after the same stale-drop.

    The two chunks are separate messages describing one action list, so a length mismatch would
    silently pair each pose with the wrong width.
    """
    node = make_node(control_hz=10.0, execute_motion=True, publish_preview=False)
    actions = _actions_with_grip([0.01 * i for i in range(8)])
    now = _t(100.0)
    t_obs = _t(99.75)  # partially stale: some leading actions get dropped

    with (
        patch.object(node._client, 'predict', return_value=ActionChunk(actions)),
        patch.object(node, 'get_clock', return_value=_FakeClock(now)),
        patch.object(node._target_pub, 'publish') as pose_pub,
        patch.object(node._gripper_pub, 'publish') as grip_pub,
    ):
        node._post_and_act(obs=_OBS, t_obs=t_obs)

    poses = pose_pub.call_args[0][0]
    grip_msg = grip_pub.call_args[0][0]
    assert 0 < len(poses) < 8  # the drop actually happened
    assert len(grip_msg.points) == len(poses)


def _diag_values(captured):
    """Reduce captured diagnostics messages to {metric: last published value}."""
    return {name: msgs[-1].data for name, msgs in captured.items() if msgs}


def _capture_diag(node):
    """Patch every diagnostics publisher to record instead of publish."""
    captured = {name: [] for name in node._diag_pubs}
    for name, pub in node._diag_pubs.items():
        pub.publish = captured[name].append
    return captured


def test_gripper_subscription_is_not_starved_by_the_image_subscription(make_node):
    """
    The gripper must not share a callback group with the 60 Hz image subscription.

    A MutuallyExclusiveCallbackGroup serialises everything in it, and 6 MB rgb8 frames at 60 Hz
    monopolise it: measured on hardware the gripper callback gapped up to 1.4 s and lost ~20% of
    samples. _gripper_width_at holds its nearest endpoint outside the buffer span, so that reaches
    the policy as a silently stale agent_pos[7] rather than as an error. Structural, and exactly
    the kind of thing a later edit re-adding the subscription would quietly undo.
    """
    node = make_node(control_hz=10.0)
    by_topic = {sub.topic_name: sub for sub in node.subscriptions}
    image_sub = by_topic['/gopro/image_raw']
    gripper_sub = by_topic['/fr3_gripper/joint_states']

    assert gripper_sub.callback_group is not image_sub.callback_group
    assert gripper_sub.callback_group is not node.default_callback_group


def test_diagnostics_report_zero_published_when_the_chunk_is_all_stale(make_node):
    """
    The zero has to reach the plot, and that is the path that returns early.

    "Nothing was commanded" is the single most useful thing on the wall — it sat silently at 0 for
    a whole session before anyone noticed the arm was not moving — so the counters are published
    before the all-stale guard, not after it.
    """
    node = make_node(control_hz=10.0, execute_motion=True, publish_preview=False)
    captured = _capture_diag(node)
    actions = _actions_with_grip([0.02] * 8)

    with (
        patch.object(node._client, 'predict', return_value=ActionChunk(actions)),
        patch.object(node, 'get_clock', return_value=_FakeClock(_t(100.0))),
    ):
        node._post_and_act(obs=_OBS, t_obs=_t(98.0))  # 2s old: stale for both devices

    values = _diag_values(captured)
    assert values['n_published_arm'] == 0
    assert values['n_published_gripper'] == 0
    assert values['n_stale_arm'] >= 8
    assert values['obs_age_s'] == pytest.approx(2.0, abs=0.01)


def test_diagnostics_report_what_each_device_actually_got(make_node):
    """The two counters must track their own slice, not a shared one."""
    node = make_node(
        control_hz=10.0,
        execute_motion=True,
        publish_preview=False,
        **{'latency.arm_exec': 0.3, 'latency.gripper_exec': 0.1},
    )
    captured = _capture_diag(node)
    actions = _actions_with_grip([0.02] * 8)

    with (
        patch.object(node._client, 'predict', return_value=ActionChunk(actions)),
        patch.object(node, 'get_clock', return_value=_FakeClock(_t(100.0))),
    ):
        node._post_and_act(obs=_OBS, t_obs=_t(99.9))

    values = _diag_values(captured)
    # 0.1s obs age: arm drops ceil(0.4/0.1)=4, gripper ceil(0.2/0.1)=2.
    assert values['n_published_arm'] == 4
    assert values['n_published_gripper'] == 6
    assert values['n_stale_arm'] == 4
    assert values['n_stale_gripper'] == 2


def test_gripper_preview_publishes_full_chunk(make_node):
    """The gripper preview mirrors the pose preview: full chunk, no execution publisher."""
    node = make_node(control_hz=10.0, publish_preview=True)  # execute_motion defaults False
    assert node._gripper_pub is None

    actions = _actions_with_grip([0.02] * 8)
    now = _t(100.0)
    with (
        patch.object(node._client, 'predict', return_value=ActionChunk(actions)),
        patch.object(node, 'get_clock', return_value=_FakeClock(now)),
        patch.object(node._gripper_preview_pub, 'publish') as grip_preview,
    ):
        node._post_and_act(obs=_OBS, t_obs=_t(98.0))  # fully stale

    grip_preview.assert_called_once()
    assert len(grip_preview.call_args[0][0].points) == 8


# ----------------------------------------------------------------------
# Where the arm chunk is aimed, and how it is anchored in time
# ----------------------------------------------------------------------
#
# The message layout itself is covered by test_target_chunk.py. What this node contributes is the
# two arguments it derives: the anchor instant, and the pre-slice index.


def test_arm_chunk_is_anchored_at_t_obs_minus_arm_exec(make_node):
    """
    The anchor carries latency.arm_exec already subtracted, so waypoints are commanded early.

    UMI does this per waypoint (`target_time - robot_action_latency` in exec_actions); the offset is
    identical for every waypoint in a chunk, so folding it into the anchor is the same thing.
    Getting the sign backwards would command every pose one arm_exec LATE, doubling the lag this
    whole path exists to remove.

    first_index must be the index in the ORIGINAL chunk, before the stale-drop slice — the poses are
    sliced but their timeline is not, so the two have to be passed separately.
    """
    node = make_node(control_hz=10.0, execute_motion=True, publish_preview=False, **{'latency.arm_exec': 0.3})
    actions = _actions_with_grip([0.02] * 8)

    with (
        patch.object(node._client, 'predict', return_value=ActionChunk(actions)),
        patch.object(node, 'get_clock', return_value=_FakeClock(_t(100.0))),
        patch.object(node._target_pub, 'publish') as pose_pub,
    ):
        node._post_and_act(obs=_OBS, t_obs=_t(99.9))

    kwargs = pose_pub.call_args[1]
    stamp = kwargs['stamp'].sec + kwargs['stamp'].nanosec * 1e-9
    assert stamp == pytest.approx(99.6)  # 99.9 - 0.3
    # 0.1s obs age + 0.3s arm_exec over a 0.1s action_dt drops 4, so the survivors start at 4.
    assert kwargs['first_index'] == 4
    assert len(pose_pub.call_args[0][0]) == 4


# ----------------------------------------------------------------------
# Wire payload and the inference worker
# ----------------------------------------------------------------------


def _image_msg(width=64, height=48, encoding='rgb8'):
    """Build a synthetic sensor_msgs/Image with a deterministic gradient."""
    frame = np.tile(np.arange(width, dtype=np.uint8)[None, :, None], (height, 1, 3))
    msg = Image()
    msg.height, msg.width, msg.encoding, msg.step = height, width, encoding, width * 3
    msg.data = frame.tobytes()
    msg.header.stamp = _t(100.0).to_msg()
    return msg


def test_cached_frame_stays_uint8(make_node):
    """
    The wire carries uint8, not float32.

    Widening to float32 before sending quadruples the request for no extra information — the
    dataset stores camera0_rgb as uint8 — and the request is bandwidth-bound.
    """
    node = make_node(control_hz=10.0, image_width=32, image_height=32)
    node._image_cb(_image_msg())

    assert node._latest_image is not None
    assert node._latest_image.dtype == np.uint8
    assert node._latest_image.shape == (32, 32, 3)


def test_tick_packs_the_channels_the_policy_needs(make_node):
    """
    The tick must emit a frame carrying camera0_rgb as uint8 and agent_pos, dt-spaced.

    Channel names are the dataset's, not the wire's own invention, so that adding a modality is
    adding a name rather than a name plus a mapping on the far side.
    """
    node = make_node(control_hz=10.0, n_obs_steps=2, image_width=224, image_height=224)
    now = _t(100.0)
    node._latest_image = np.zeros((224, 224, 3), dtype=np.uint8)
    node._latest_image_stamp = now
    node._episode_reset_done = True
    for _ in range(node._n_obs_steps):
        node._obs_buffer.append((node._latest_image, np.zeros(8)))

    captured = []
    with (
        patch.object(node, 'get_clock', return_value=_FakeClock(now)),
        patch.object(node, '_lookup_agent_pos', return_value=np.zeros(8)),
        patch.object(node, '_submit_inference', side_effect=lambda b, t: captured.append(b)),
    ):
        node._control_tick()

    assert len(captured) == 1
    obs = captured[0]

    assert obs.names() == ['agent_pos', 'camera0_rgb']
    assert obs['camera0_rgb'].dtype == np.uint8
    assert obs['camera0_rgb'].shape == (2, 224, 224, 3)
    assert obs['agent_pos'].shape == (2, 8)
    assert obs.n_obs_steps == 2
    assert obs.n_action_steps == node._n_action_steps
    # Raw bytes plus a small header. The float32-and-base64 form this replaced was 1.6 MB; the
    # bound is loose on purpose so it survives a header formatting change.
    assert len(obs.to_frame()) < 350_000


def test_submit_keeps_only_the_newest_observation(make_node):
    """
    A superseded observation is dropped, never queued.

    _post_and_act only ever sees the newest observation still pending once the worker frees up.
    By the time a backlog could deliver a queued one, _n_stale_actions would discard every action
    the chunk contained — so queueing would buy a round trip's worth of GPU time for nothing.
    """
    node = make_node(control_hz=10.0)
    first_started = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    sent = []

    def _slow_post(payload, t_obs):
        sent.append(payload)
        if payload == {'in_flight': True}:
            first_started.set()
            release_first.wait(timeout=5.0)
        else:
            second_done.set()

    with patch.object(node, '_post_and_act', side_effect=_slow_post):
        node._submit_inference({'in_flight': True}, _t(100.0))
        assert first_started.wait(timeout=5.0), 'worker never picked up the observation'

        # Both land while the worker is still busy with 'in_flight'; only the newest one may
        # still be waiting when it frees up.
        node._submit_inference({'first': True}, _t(100.1))
        node._submit_inference({'second': True}, _t(100.2))

        release_first.set()
        assert second_done.wait(timeout=5.0), 'worker never picked up the newest observation'

    assert sent == [{'in_flight': True}, {'second': True}]


def test_tick_does_not_block_on_the_request(make_node):
    """The control tick hands off and returns; the request runs on the worker thread, not here."""
    node = make_node(control_hz=10.0)
    started = threading.Event()
    release = threading.Event()

    def _slow_post(payload, t_obs):
        started.set()
        release.wait(timeout=5.0)

    with patch.object(node, '_post_and_act', side_effect=_slow_post):
        node._submit_inference({}, _t(100.0))
        assert started.wait(timeout=5.0), 'worker never picked up the observation'
        # The worker is mid-"request". A tick submitting now must return immediately.
        t0 = time.monotonic()
        node._submit_inference({}, _t(100.1))
        assert time.monotonic() - t0 < 0.5
        release.set()


def test_control_tick_does_not_block_on_reset(make_node):
    """
    The episode-start /reset also runs on the worker thread, not the control tick.

    A tick that dispatches it inline would stall the whole loop (buffer included) for up to
    post_timeout_s on every retry while the server is slow or unreachable — the same failure
    mode _submit_inference exists to avoid for /predict_cartesian/.
    """
    node = make_node(control_hz=10.0)
    now = _t(100.0)
    node._latest_image = np.zeros((4, 4, 3), dtype=np.uint8)
    node._latest_image_stamp = now
    for _ in range(node._n_obs_steps):
        node._obs_buffer.append((node._latest_image, np.zeros(8)))

    started = threading.Event()
    release = threading.Event()

    def _slow_reset(agent_pos):
        started.set()
        release.wait(timeout=5.0)

    with (
        patch.object(node, 'get_clock', return_value=_FakeClock(now)),
        patch.object(node, '_lookup_agent_pos', return_value=np.zeros(8)),
        patch.object(node._client, 'reset', side_effect=_slow_reset),
    ):
        t0 = time.monotonic()
        node._control_tick()
        assert time.monotonic() - t0 < 0.5, '_control_tick blocked on the reset request'
        assert started.wait(timeout=5.0), 'worker never picked up the reset'
        release.set()


def test_worker_does_a_pending_reset_before_a_pending_observation(make_node):
    """A late reset makes every wrt_start pose in the episode wrong; a late inference tick doesn't."""
    node = make_node(control_hz=10.0)
    order = []
    reset_started = threading.Event()
    release_reset = threading.Event()

    def _slow_reset(agent_pos):
        reset_started.set()
        release_reset.wait(timeout=5.0)
        order.append('reset')

    with (
        patch.object(node, '_post_and_act', side_effect=lambda *a, **k: order.append('predict')),
        patch.object(node._client, 'reset', side_effect=_slow_reset),
    ):
        node._submit_reset(np.zeros(8))
        assert reset_started.wait(timeout=5.0), 'worker never picked up the reset'
        # The worker is now blocked inside the reset call; submitting an observation here can
        # only land in the mailbox, never jump ahead of it.
        node._submit_inference(_OBS, _t(100.0))
        release_reset.set()

        for _ in range(50):
            if len(order) >= 2:
                break
            time.sleep(0.05)

    assert order == ['reset', 'predict']

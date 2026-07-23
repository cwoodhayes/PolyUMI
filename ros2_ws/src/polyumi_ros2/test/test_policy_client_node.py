"""
Tests for policy_client_node's latency math and configuration validation.

These cover the time arithmetic that decides *which* pose gets paired with an image and *how
much* of an action chunk is still valid. Both are easy to get subtly wrong in a way no runtime
error reveals — a sign flip in the pose lookup just trains/deploys on a slightly-off pose, and
a bad stale count just moves the arm to the wrong waypoint — so they are pinned here rather
than left to hardware testing to notice.
"""

from unittest.mock import patch

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.clock import ClockType
from rclpy.parameter import Parameter
from rclpy.time import Time

from polyumi_ros2.policy_client_node import PolicyClientNode

BASE_FRAME = 'fr3_link0'
EEF_FRAME = 'fr3_hand_tcp'


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
    """The values actually shipped in config/latency.yaml must pass validation."""
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


def _n_stale_at(node, *, obs_age_s: float) -> int:
    """Evaluate _n_stale_actions for an observation captured obs_age_s ago (may be negative)."""
    now = _t(100.0)
    t_obs = _t(100.0 - obs_age_s)
    with patch.object(node, 'get_clock', return_value=_FakeClock(now)):
        return node._n_stale_actions(t_obs)


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


def test_reset_episode_posts_start_pose_once(make_node):
    """_reset_episode POSTs the given pose to the reset URL and latches _episode_reset_done."""
    node = make_node(inference_server_url='http://sheep:8000/predict_cartesian/')
    agent_pos = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0])

    with patch.object(node, '_http_post_json', return_value={'status': 'ok'}) as post:
        node._reset_episode(agent_pos)

    assert node._episode_reset_done is True
    post.assert_called_once()
    url, body = post.call_args[0]
    assert url == 'http://sheep:8000/reset'
    assert body == {'agent_pos': agent_pos.tolist()}


def test_reset_episode_not_latched_on_failure(make_node):
    """A failed /reset leaves the flag unset so the next tick retries."""
    node = make_node()
    with patch.object(node, '_http_post_json', return_value=None):
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
    with patch.object(node, '_http_post_json', return_value={'actions': actions}), \
            patch.object(node, 'get_clock', return_value=_FakeClock(now)), \
            patch.object(node._preview_pub, 'publish') as preview_pub:
        node._post_and_act(payload={}, t_obs=t_obs)

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


def test_tf_use_latest_ignores_stamp(make_node):
    """tf_use_latest looks up the newest transform regardless of the requested image stamp."""
    node = make_node(tf_use_latest=True, **{'latency.gopro': 0.05, 'buffers.ee_pose_s': 1.0})
    _push_ramp_tf(node)  # ramp x==timestamp over [0.6, 1.4]

    # A stamp well before the ramp would extrapolate-into-past in time-aligned mode; with
    # tf_use_latest it returns the newest sample (x == 1.4) instead.
    agent_pos = node._lookup_agent_pos(image_stamp=_t(0.1))
    assert agent_pos is not None
    assert agent_pos[0] == pytest.approx(1.4, abs=1e-6)

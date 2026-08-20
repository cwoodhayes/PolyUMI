"""
Tests for the NUC-side gripper bridge's command loop.

This is where every stateful decision on the gripper path lives — the rate limit, the deadband,
and the _last_commanded claim it measures against — and all of it fails *quietly*: a rate limit
that never releases just stops commanding, a _last_commanded left claiming a width the hand never
got just parks the hand there. Nothing raises, so nothing shows up in a log except an absence.

Runs on the laptop despite the bridge targeting the Humble NUC: only the franka_msgs *message
definitions* are needed, and those are built in ros2_ws. No action servers are involved — the
ActionClient is mocked out, so these tests exercise the bridge's logic and nothing else.

    bash -c 'unset VIRTUAL_ENV; source /opt/ros/kilted/setup.bash \
      && source ros2_ws/install/setup.bash \
      && /usr/bin/python3 -m pytest nuc/test_fr3_gripper_bridge.py -q'
"""

from unittest.mock import MagicMock, patch

import pytest
import rclpy
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import fr3_gripper_bridge as gb


@pytest.fixture(scope='module', autouse=True)
def ros():
    """Init rclpy once for the module; every node here is constructed without a real executor."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def make_node():
    """
    Build a bridge with mocked action clients, so no server or executor is required.

    wait_for_server/server_is_ready both answer True: the tests that care about an absent server
    override them explicitly, and the rest should not have to.
    """
    nodes = []

    def _make(**overrides):
        params = [Parameter(k, value=v) for k, v in overrides.items()]
        with patch.object(gb, 'ActionClient') as action_client:
            action_client.return_value.wait_for_server.return_value = True
            action_client.return_value.server_is_ready.return_value = True
            node = gb.Fr3GripperBridge(parameter_overrides=params)
        node.get_logger = MagicMock()
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node.destroy_node()


def _chunk(widths, dt=0.1) -> JointTrajectory:
    """Build a gripper chunk as policy_client_node publishes one: one point per action step."""
    msg = JointTrajectory()
    msg.joint_names = ['fr3_gripper_width']
    for i, width in enumerate(widths):
        point = JointTrajectoryPoint()
        point.positions = [width]
        point.time_from_start = Duration(seconds=i * dt).to_msg()
        msg.points.append(point)
    return msg


def _capture_sends(node) -> list:
    """Replace _send with a recorder, returning the list it appends (label, goal, width) to."""
    sent = []
    node._send = lambda client, goal, label, width: sent.append((label, goal, width))
    return sent


def _allow_send(node) -> None:
    """Clear the rate limit so the next _tick may command, isolating the logic under test."""
    node._last_sent_at = None


def _push_state(node, width_m: float) -> None:
    """Feed one gripper joint state, split across the two fingers as the FR3 reports it."""
    msg = JointState()
    msg.position = [width_m / 2.0, width_m / 2.0]
    node._on_state(msg)


# ----------------------------------------------------------------------
# Target intake
# ----------------------------------------------------------------------


def test_state_sums_the_two_finger_joints(make_node):
    """Aperture is the SUM of the finger positions — each reports half the opening."""
    node = make_node()
    _push_state(node, 0.06)

    assert node._current_width == pytest.approx(0.06)


def test_the_first_chunk_point_is_the_one_tracked(make_node):
    """
    No lead here: policy_client_node already truncated this chunk by latency.gripper_exec.

    A gripper_lead_steps parameter used to index further into the chunk, compensating for the
    chunk having been truncated by the ARM's latency instead of the hand's. Each device now gets
    its own slice, so point 0 is already the width intended for when the fingers start moving, and
    any lead here would double-compensate.
    """
    node = make_node()
    node._on_target(_chunk([0.01, 0.02, 0.03, 0.04]))
    assert node._desired_width == pytest.approx(0.01)


def test_target_width_is_clamped_to_the_robot_range(make_node):
    """The handheld gripper opens wider than the hand, and a negative width is meaningless."""
    node = make_node(max_width_m=0.08)
    node._on_target(_chunk([0.5]))
    assert node._desired_width == pytest.approx(0.08)

    node._on_target(_chunk([-0.2]))
    assert node._desired_width == pytest.approx(0.0)


def test_degenerate_chunks_are_ignored(make_node):
    """An empty chunk, or a point with no positions, leaves the last target standing."""
    node = make_node()
    node._on_target(_chunk([0.03]))

    node._on_target(JointTrajectory())
    empty_point = JointTrajectory()
    empty_point.points.append(JointTrajectoryPoint())
    node._on_target(empty_point)

    assert node._desired_width == pytest.approx(0.03)


# ----------------------------------------------------------------------
# Command loop: the rate limit and the deadband
# ----------------------------------------------------------------------


def test_deadband_suppresses_a_repeat_command(make_node):
    """
    A target that has not moved past the deadband issues nothing.

    This is the whole point of the rate limiting: franka_gripper accepts every goal into a
    detached thread with no queue, so an unfiltered 10 Hz stream is ~9 aborted goals a second.
    """
    node = make_node(execute=False, width_deadband_m=0.005)
    node._on_target(_chunk([0.05]))
    node._tick()
    node._on_target(_chunk([0.052]))  # 2mm — inside the deadband
    _allow_send(node)  # so it is the deadband being tested, not the rate limit
    node._tick()

    logged = [c for c in node.get_logger().info.call_args_list if 'log-only' in str(c)]
    assert len(logged) == 1


def test_a_move_past_the_deadband_commands_again(make_node):
    """Once the target really has moved, the next tick commands."""
    node = make_node(execute=False, width_deadband_m=0.005)
    node._on_target(_chunk([0.05]))
    node._tick()
    node._on_target(_chunk([0.07]))
    _allow_send(node)
    node._tick()

    logged = [c for c in node.get_logger().info.call_args_list if 'log-only' in str(c)]
    assert len(logged) == 2


def test_the_rate_limit_holds_off_the_next_command(make_node):
    """A second tick inside min_command_period_s sends nothing, however far the target moved."""
    node = make_node(execute=True, min_command_period_s=0.25)
    sent = _capture_sends(node)
    node._on_target(_chunk([0.05]))

    node._tick()
    node._on_target(_chunk([0.02]))  # well past the deadband
    node._tick()

    assert len(sent) == 1


def test_the_rate_limit_is_not_the_tick_period(make_node):
    """
    Ticks are far cheaper than commands, so a ready target waits at most the period.

    The two used to be the same number, which meant a target that became ready just after a tick
    waited a whole period for no reason — pure quantisation on top of the hand's own delay.
    """
    node = make_node(min_command_period_s=0.25)
    assert gb.TICK_PERIOD_S < node._period


def test_an_unresolved_goal_does_not_block_the_next_command(make_node):
    """
    The bridge waits on the rate limit, never on the previous goal finishing.

    A Move's result arrives only once the fingers stop, so gating on it made the hand deaf to a
    policy reversal for the whole stroke. libfranka lets a new goal supersede an unfinished one —
    the superseded command returns "Command aborted!", which _on_goal_result logs at INFO.
    """
    node = make_node(execute=True)
    sent = _capture_sends(node)  # nothing ever resolves: _send is replaced, no callbacks fire
    node._on_target(_chunk([0.05]))
    node._tick()

    node._on_target(_chunk([0.02]))
    _allow_send(node)
    node._tick()

    assert len(sent) == 2


def test_log_only_mode_still_deadbands(make_node):
    """A dry run's log must be deadbanded, not one line per tick."""
    node = make_node(execute=False)
    node._on_target(_chunk([0.05]))
    node._tick()

    assert node._last_commanded == pytest.approx(0.05)


# ----------------------------------------------------------------------
# Goal resolution: what gets committed to _last_commanded, and when
# ----------------------------------------------------------------------


def _goal_response(accepted: bool):
    """Build a fake send_goal_async future that resolves to an accepted/rejected goal handle."""
    future = MagicMock()
    handle = MagicMock()
    handle.accepted = accepted
    future.result.return_value = handle
    return future, handle


def test_an_accepted_goal_keeps_the_deadband_reference(make_node):
    """The width _tick claimed stays claimed once the hand has it."""
    node = make_node(execute=True)
    node._last_commanded = 0.05
    future, _ = _goal_response(accepted=True)

    node._on_goal_response(future, 'Move', width=0.05)

    assert node._last_commanded == pytest.approx(0.05)


def test_a_rejected_goal_stays_retryable(make_node):
    """
    A goal the hand never got must not be deadbanded away.

    Leaving _last_commanded claimed would suppress every retry until the policy's target drifted
    a full deadband, parking the hand at a width it was never told about.
    """
    node = make_node(execute=True)
    node._last_commanded = 0.05
    future, _ = _goal_response(accepted=False)

    node._on_goal_response(future, 'Move', width=0.05)

    assert node._last_commanded is None


def test_a_late_rejection_does_not_unclaim_a_newer_width(make_node):
    """
    Goals resolve out of order under preemption, so unclaiming must match on the width.

    Otherwise a rejection arriving after the next goal was already accepted drops that newer
    claim, and the bridge re-sends a width the hand is already holding.
    """
    node = make_node(execute=True)
    node._last_commanded = 0.02  # a newer goal has since been sent and claimed
    future, _ = _goal_response(accepted=False)

    node._on_goal_response(future, 'Move', width=0.05)  # the OLD one, rejected late

    assert node._last_commanded == pytest.approx(0.02)


def test_a_send_failure_stays_retryable(make_node):
    """
    A send whose future carries an exception must unclaim, not escape the callback.

    rclpy's Future.result() re-raises what the send stored rather than returning None. Unguarded,
    the raise escapes into rclpy internals and _last_commanded keeps claiming a width the hand
    never got — which the deadband then suppresses every retry against.
    """
    node = make_node(execute=True)
    node._last_commanded = 0.05
    future = MagicMock()
    future.result.side_effect = RuntimeError('send failed')

    node._on_goal_response(future, 'Move', width=0.05)

    assert node._last_commanded is None


def test_a_raising_result_future_does_not_escape(make_node):
    """Same guard on the result half; there is nothing to release, only a raise to swallow."""
    node = make_node(execute=True)
    future = MagicMock()
    future.result.side_effect = RuntimeError('result failed')

    node._on_goal_result(future, 'Move')  # must not raise


def test_an_unavailable_server_drops_the_goal_and_stays_retryable(make_node):
    """A dropped goal has to leave the bridge able to retry, not holding the claim."""
    node = make_node(execute=True)
    node._last_commanded = 0.05
    node._move.server_is_ready.return_value = False

    node._send(node._move, gb.Move.Goal(), 'Move', width=0.05)

    assert node._last_commanded is None


# ----------------------------------------------------------------------
# Action selection and speed
# ----------------------------------------------------------------------


def test_speed_is_derived_from_distance_and_the_chunk_timeline(make_node):
    """Speed sizes to the move, clamped into the configured band at both ends."""
    node = make_node(min_speed_mps=0.02, max_speed_mps=0.15)

    assert node._desired_speed(0.05, 0.04, 0.25) == pytest.approx(0.04)  # 0.01m / 0.25s
    assert node._desired_speed(0.041, 0.04, 0.25) == pytest.approx(0.02)  # clamped to min
    assert node._desired_speed(0.08, 0.0, 0.25) == pytest.approx(0.15)  # clamped to max


def test_speed_falls_back_to_the_maximum_without_a_state_reading(make_node):
    """With no measured width there is no distance to size against."""
    node = make_node(max_speed_mps=0.15)

    assert node._desired_speed(0.05, None, 0.25) == pytest.approx(0.15)


def test_grasp_is_used_only_when_closing_below_the_threshold(make_node):
    """
    Grasp (force-controlled) applies only on a CLOSING move below the threshold.

    Move applies no force, so opening or moving above the threshold must stay a Move — a Grasp
    on an opening command would squeeze against nothing.
    """
    node = make_node(execute=True, use_grasp_below_m=0.03, width_deadband_m=0.001)
    sent = _capture_sends(node)

    _push_state(node, 0.06)
    node._on_target(_chunk([0.02]))  # closing, below the threshold
    node._tick()
    assert sent[-1][0] == 'Grasp'

    _allow_send(node)
    _push_state(node, 0.01)
    node._on_target(_chunk([0.025]))  # below the threshold but OPENING
    node._tick()
    assert sent[-1][0] == 'Move'


def test_grasp_is_disabled_by_default(make_node):
    """
    The shipped default is Move-only: bringup moves fingers and applies no force.

    use_grasp_below_m defaults to 0.0 precisely so a first hardware run cannot squeeze anything.
    """
    node = make_node(execute=True)
    sent = _capture_sends(node)

    _push_state(node, 0.06)
    node._on_target(_chunk([0.0]))
    node._tick()

    assert sent[-1][0] == 'Move'


# ----------------------------------------------------------------------
# Parameter validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    'bad',
    [
        {'min_command_period_s': 0.0},  # divides by zero inside _desired_speed
        {'min_speed_mps': 0.0},  # issues goals the hand never completes
        {'max_speed_mps': 0.01},  # inverted range: the clamp returns the wrong bound
        {'width_deadband_m': -0.001},
        {'max_width_m': 0.0},
        {'use_grasp_below_m': -0.01},
        {'use_grasp_below_m': 0.03, 'grasp_force_n': 0.0},
        {'grasp_epsilon_m': -0.01},
    ],
)
def test_invalid_parameters_fail_fast(make_node, bad):
    """
    Bad configuration raises at construction rather than degrading inside a timer callback.

    The sharp one is min_command_period_s: a zero divides by zero in _desired_speed, inside a
    timer callback, where the traceback scrolls past and the node then sits commanding nothing.
    """
    with pytest.raises(ValueError, match='Invalid fr3_gripper_bridge configuration'):
        make_node(**bad)


def test_deadband_hold_is_logged_once_with_the_actual_width(make_node):
    """
    A stall must be visible in the log, and must name the hand's real width.

    The bridge only logs when it sends, so without this a suppressing deadband is
    indistinguishable from no chunks arriving. The actual width is what tells the two apart:
    if it has drifted from _last_commanded, the hand never reached what it accepted.
    """
    node = make_node(execute=True)
    node._last_commanded = 0.050
    node._current_width = 0.070  # hand stalled 20mm away from what it accepted
    node._desired_width = 0.052  # inside the 5mm deadband -> suppressed

    node._tick()
    node._tick()

    holds = [c for c in node.get_logger().info.call_args_list if 'Holding' in str(c)]
    assert len(holds) == 1, 'the hold must be logged exactly once per stall, not every period'
    message = str(holds[0])
    assert '0.0500' in message, 'must report what was last commanded'
    assert '0.0700' in message, 'must report where the hand actually is'


def test_deadband_hold_logs_again_after_the_target_moves_away(make_node):
    """The latch has to reset, or a second stall in the same run goes unreported."""
    node = make_node(execute=True)
    node._last_commanded = 0.050
    node._current_width = 0.050
    node._desired_width = 0.052
    node._tick()

    node._desired_width = 0.070  # past the deadband: sends, clearing the latch
    node._tick()
    _allow_send(node)
    node._last_commanded = 0.070
    node._desired_width = 0.072  # inside the deadband again
    node._tick()

    holds = [c for c in node.get_logger().info.call_args_list if 'Holding' in str(c)]
    assert len(holds) == 2

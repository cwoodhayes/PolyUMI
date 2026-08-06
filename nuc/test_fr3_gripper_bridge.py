"""
Tests for the NUC-side gripper bridge's command loop.

This is where every stateful decision on the gripper path lives — the single in-flight goal
slot, the watchdog that frees it, the deadband, and the deferred _last_commanded commit — and
all of it fails *quietly*: a wedged slot just stops commanding, a mis-committed _last_commanded
just parks the hand at a width it never received. Nothing raises, so nothing shows up in a log
except an absence.

Runs on the laptop despite the bridge targeting the Humble NUC: only the franka_msgs *message
definitions* are needed, and those are built in ros2_ws. No action servers are involved — the
ActionClient is mocked out, so these tests exercise the bridge's logic and nothing else.

    bash -c 'unset VIRTUAL_ENV; source /opt/ros/kilted/setup.bash \
      && source ros2_ws/install/setup.bash \
      && /usr/bin/python3 -m pytest nuc/test_fr3_gripper_bridge.py -q'
"""

import time
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


def test_lead_steps_selects_a_later_point_and_clamps(make_node):
    """gripper_lead_steps indexes into the chunk, saturating at the last point rather than raising."""
    node = make_node(gripper_lead_steps=2)
    node._on_target(_chunk([0.01, 0.02, 0.03, 0.04]))
    assert node._desired_width == pytest.approx(0.03)

    far = make_node(gripper_lead_steps=99)
    far._on_target(_chunk([0.01, 0.02, 0.03]))
    assert far._desired_width == pytest.approx(0.03)


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
# Command loop: deadband and the in-flight slot
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
    node._tick()

    logged = [c for c in node.get_logger().info.call_args_list if 'log-only' in str(c)]
    assert len(logged) == 1


def test_a_move_past_the_deadband_commands_again(make_node):
    """Once the target really has moved, the next tick commands."""
    node = make_node(execute=False, width_deadband_m=0.005)
    node._on_target(_chunk([0.05]))
    node._tick()
    node._on_target(_chunk([0.07]))
    node._tick()

    logged = [c for c in node.get_logger().info.call_args_list if 'log-only' in str(c)]
    assert len(logged) == 2


def test_only_one_goal_is_in_flight_at_a_time(make_node):
    """A second tick while a goal is unresolved sends nothing — the slot is claimed."""
    node = make_node(execute=True)
    sent = _capture_sends(node)
    node._on_target(_chunk([0.05]))

    node._tick()
    node._on_target(_chunk([0.02]))  # well past the deadband
    node._tick()

    assert len(sent) == 1


def test_watchdog_frees_a_slot_whose_goal_never_resolved(make_node):
    """
    A goal that never resolves must not wedge the bridge permanently.

    Goal resolution is callback-driven, so a lost result callback (server died mid-motion) would
    otherwise stop the bridge commanding forever, and silently.
    """
    node = make_node(execute=True)
    sent = _capture_sends(node)
    node._on_target(_chunk([0.05]))
    node._tick()
    assert node._goal_in_flight

    node._goal_sent_at = time.monotonic() - (gb.GOAL_RESULT_TIMEOUT_S + 1.0)
    node._tick()  # notices the timeout, frees the slot, commands on the NEXT tick
    assert not node._goal_in_flight
    assert len(sent) == 1

    node._tick()
    assert len(sent) == 2


def test_log_only_mode_never_claims_the_slot(make_node):
    """A dry run must keep logging every distinct width, not stall behind a phantom goal."""
    node = make_node(execute=False)
    node._on_target(_chunk([0.05]))
    node._tick()

    assert not node._goal_in_flight
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


def test_last_commanded_is_committed_only_on_acceptance(make_node):
    """The deadband measures drift from what the hand actually accepted, not what we attempted."""
    node = make_node(execute=True)
    node._goal_in_flight = True
    future, _ = _goal_response(accepted=True)

    node._on_goal_response(future, 'Move', width=0.05)

    assert node._last_commanded == pytest.approx(0.05)


def test_a_rejected_goal_stays_retryable(make_node):
    """
    A goal the hand never got must not be deadbanded away.

    Committing _last_commanded on send would suppress every retry until the policy's target
    drifted a full deadband, parking the hand at a width it was never told about.
    """
    node = make_node(execute=True)
    node._goal_in_flight = True
    future, _ = _goal_response(accepted=False)

    node._on_goal_response(future, 'Move', width=0.05)

    assert node._last_commanded is None
    assert not node._goal_in_flight


def test_a_send_failure_frees_the_slot_instead_of_wedging_it(make_node):
    """
    A send whose future carries an exception must free the slot, not escape the callback.

    rclpy's Future.result() re-raises what the send stored rather than returning None.

    Unguarded, the raise escapes into rclpy internals and _clear_in_flight never runs — the
    bridge then stops commanding for a full GOAL_RESULT_TIMEOUT_S, with nothing in the log to
    say why.
    """
    node = make_node(execute=True)
    node._goal_in_flight = True
    future = MagicMock()
    future.result.side_effect = RuntimeError('send failed')

    node._on_goal_response(future, 'Move', width=0.05)

    assert not node._goal_in_flight
    assert node._last_commanded is None


def test_a_raising_result_future_frees_the_slot_too(make_node):
    """Same guarantee on the result half of the handshake."""
    node = make_node(execute=True)
    node._goal_in_flight = True
    future = MagicMock()
    future.result.side_effect = RuntimeError('result failed')

    node._on_goal_result(future, 'Move')

    assert not node._goal_in_flight


def test_an_unavailable_server_drops_the_goal_and_frees_the_slot(make_node):
    """A dropped goal has to leave the bridge able to retry, not holding a claimed slot."""
    node = make_node(execute=True)
    node._goal_in_flight = True
    node._move.server_is_ready.return_value = False

    node._send(node._move, gb.Move.Goal(), 'Move', width=0.05)

    assert not node._goal_in_flight
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

    node._clear_in_flight()
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
        {'gripper_lead_steps': -1},
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

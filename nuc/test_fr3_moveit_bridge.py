"""
Tests for the NUC-side MoveIt bridge's /polyumi/home service.

Homing is the one path here that moves the arm on an explicit request rather than on a streamed
chunk, and every way it can be wrong is silent: a wrong joint name plans to a pose that is not
home, the plan-only gate would make it a no-op in the default configuration, and the shorter
chunk timeout would abort a long joint-space sweep partway across the workspace.

Runs on the laptop despite the bridge targeting the Humble NUC: only the moveit_msgs *message
definitions* are needed. No move_group is involved — the service and action clients are mocked,
so these tests exercise the bridge's logic and nothing else.

    bash -c 'unset VIRTUAL_ENV; source /opt/ros/kilted/setup.bash \
      && source ros2_ws/install/setup.bash \
      && /usr/bin/python3 -m pytest nuc/test_fr3_moveit_bridge.py -q'
"""

from unittest.mock import MagicMock, patch

from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory
import pytest
import rclpy
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

import fr3_moveit_bridge as mb


@pytest.fixture(scope='module', autouse=True)
def ros():
    """Init rclpy once for the module; every node here is constructed without a real executor."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def make_node():
    """
    Build a bridge whose move_group clients are mocks, so no server or executor is required.

    create_client is patched to hand back a fresh mock per call — the real one would make
    __init__ block for its two 10 s wait_for_service/wait_for_server timeouts.
    """
    nodes = []

    def _make(**overrides):
        params = [Parameter(k, value=v) for k, v in overrides.items()]
        with (
            patch.object(mb.Fr3MoveItBridge, 'create_client', side_effect=lambda *a, **k: MagicMock()),
            patch.object(mb, 'ActionClient') as action_client,
        ):
            action_client.return_value.wait_for_server.return_value = True
            action_client.return_value.server_is_ready.return_value = True
            node = mb.Fr3MoveItBridge(parameter_overrides=params)
        node.get_logger = MagicMock()
        # Futures never resolve against a mocked client, so _wait would burn its full timeout.
        node._wait = lambda future, timeout_s: True
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node.destroy_node()


def _stub_plan_ok(node, trajectory='TRAJ'):
    """Make the joint-space planner answer SUCCESS with `trajectory`."""
    node._joint_plan.service_is_ready.return_value = True
    resp = MagicMock()
    resp.motion_plan_response.error_code.val = MoveItErrorCodes.SUCCESS
    resp.motion_plan_response.trajectory = trajectory
    node._joint_plan.call_async.return_value.result.return_value = resp
    return resp


def _capture_execute(node) -> list:
    """Replace _run_execute with a recorder, returning the list it appends (trajectory, timeout) to."""
    calls = []

    def _record(trajectory, timeout_s):
        calls.append((trajectory, timeout_s))
        return True

    node._run_execute = _record
    return calls


def _home(node) -> Trigger.Response:
    """Call the service handler directly, as rclpy would."""
    return node._on_home(Trigger.Request(), Trigger.Response())


def test_home_plans_to_the_srdf_ready_pose(make_node):
    """The goal constraint must name fr3_joint1..7 at the SRDF `ready` values, in order."""
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)

    assert _home(node).success

    request = node._joint_plan.call_async.call_args[0][0]
    constraints = request.motion_plan_request.goal_constraints[0].joint_constraints
    assert [c.joint_name for c in constraints] == [f'fr3_joint{i}' for i in range(1, 8)]
    assert [c.position for c in constraints] == pytest.approx(mb.HOME_JOINTS)
    assert request.motion_plan_request.group_name == mb.DEFAULT_GROUP


def test_home_moves_the_arm_even_in_plan_only_mode(make_node):
    """`execute` gates streamed chunks, not an explicit home request — see _on_home's docstring."""
    node = make_node(execute=False)
    _stub_plan_ok(node)
    executed = _capture_execute(node)

    assert _home(node).success
    assert len(executed) == 1, 'plan-only mode must not suppress an explicit /polyumi/home'


def test_home_uses_the_long_execute_timeout(make_node):
    """A joint-space sweep at low velocity scaling needs far longer than a short move."""
    node = make_node()
    _stub_plan_ok(node)
    executed = _capture_execute(node)

    _home(node)

    assert executed[0][1] == mb.HOME_EXECUTE_TIMEOUT_S


def test_home_refused_while_a_chunk_is_in_flight(make_node):
    """A concurrent /polyumi/home call must not cut in on one already planning/executing."""
    node = make_node()
    _stub_plan_ok(node)
    executed = _capture_execute(node)
    node._busy.acquire()
    try:
        response = _home(node)
    finally:
        node._busy.release()

    assert not response.success
    assert 'busy' in response.message
    assert executed == []


def test_home_releases_the_busy_lock_after_a_failed_plan(make_node):
    """A planning failure must not wedge the bridge — the next chunk still has to get through."""
    node = make_node()
    node._joint_plan.service_is_ready.return_value = False  # move_group missing
    _capture_execute(node)

    assert not _home(node).success
    assert node._busy.acquire(blocking=False), 'busy lock left held after a failed home'
    node._busy.release()


def test_home_rejects_a_wrong_length_home_joints(make_node):
    """A 6-value override would otherwise zip() short and silently home to a partial pose."""
    node = make_node(home_joints=[0.0] * 6)
    _stub_plan_ok(node)
    executed = _capture_execute(node)

    response = _home(node)

    assert not response.success
    assert 'expected 7' in response.message
    assert executed == []


def test_execute_timeout_cancels_the_goal_and_waits_for_the_abort(make_node):
    """
    A timed-out goal keeps driving the arm, and the caller releases the busy lock on the way out.

    Without a cancel the next chunk plans a Cartesian path from a start state the arm has already
    left. The wait after the cancel matters just as much: returning while it decelerates puts us
    back in the same race, one deceleration narrower.
    """
    node = make_node(execute=True)
    waits = []

    def _wait(_future, timeout_s):
        """Time out only on the second wait — the goal result — and succeed on the others."""
        waits.append(timeout_s)
        return len(waits) != 2

    node._wait = _wait

    assert not node._run_execute(RobotTrajectory(), timeout_s=7.0)

    goal_handle = node._exec.send_goal_async.return_value.result.return_value
    goal_handle.cancel_goal_async.assert_called_once()
    assert waits == [mb.PLAN_TIMEOUT_S, 7.0, mb.CANCEL_TIMEOUT_S], (
        'must wait for the abort to land, bounded by CANCEL_TIMEOUT_S'
    )


def test_execute_success_does_not_cancel(make_node):
    """The cancel path must not fire on the happy path — it would abort every good chunk."""
    node = make_node(execute=True)
    result = node._exec.send_goal_async.return_value.result.return_value.get_result_async.return_value
    result.result.return_value.result.error_code.val = MoveItErrorCodes.SUCCESS

    assert node._run_execute(RobotTrajectory(), timeout_s=7.0)

    node._exec.send_goal_async.return_value.result.return_value.cancel_goal_async.assert_not_called()


# ----------------------------------------------------------------------
# Controller handover
# ----------------------------------------------------------------------


def _stub_switch(node, ok=True):
    """Make /controller_manager/switch_controller answer `ok`, and record the requests it got."""
    node._switch.wait_for_service.return_value = True
    result = MagicMock()
    result.ok = ok
    node._switch.call_async.return_value.result.return_value = result
    return node._switch.call_async


def _switch_pairs(call_async) -> list:
    """Return the (activate, deactivate) pairs the bridge asked for, in order."""
    return [
        (tuple(c[0][0].activate_controllers), tuple(c[0][0].deactivate_controllers)) for c in call_async.call_args_list
    ]


def _stub_switch_sequence(node, oks: list[bool]):
    """Like _stub_switch, but each successive switch_controller call answers the next `oks` value."""
    node._switch.wait_for_service.return_value = True
    futures = []
    for ok in oks:
        result = MagicMock()
        result.ok = ok
        future = MagicMock()
        future.result.return_value = result
        futures.append(future)
    node._switch.call_async.side_effect = futures
    return node._switch.call_async


def test_home_borrows_the_arm_from_the_servo_and_gives_it_back(make_node):
    """
    Homing must borrow the arm from the streaming controller and then give it back.

    Homing goes through move_group, which drives fr3_arm_controller; the streaming impedance
    controller claims the same effort interfaces, so exactly one can hold the arm. Failing to hand
    it back would leave the policy silently commanding nothing after every home.
    """
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)
    call_async = _stub_switch(node)

    assert _home(node).success

    assert _switch_pairs(call_async) == [
        ((mb.MOVEIT_CONTROLLER,), (mb.SERVO_CONTROLLER,)),
        ((mb.SERVO_CONTROLLER,), (mb.MOVEIT_CONTROLLER,)),
    ]


def test_home_switches_strictly(make_node):
    """
    The handover must be STRICT.

    BEST_EFFORT could half-apply, leaving the arm with two controllers claiming its efforts or
    none — and "none" looks exactly like a working system that has stopped moving.
    """
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)
    call_async = _stub_switch(node)

    _home(node)

    request = call_async.call_args_list[0][0][0]
    assert request.strictness == mb.SwitchController.Request.STRICT


def test_home_hands_the_arm_back_even_when_planning_fails(make_node):
    """A failed home must not strand the servo deactivated — that is a silent dead policy."""
    node = make_node()
    node._joint_plan.service_is_ready.return_value = False  # planner unavailable
    call_async = _stub_switch(node)

    assert not _home(node).success

    assert _switch_pairs(call_async)[-1] == ((mb.SERVO_CONTROLLER,), (mb.MOVEIT_CONTROLLER,))


def test_home_reports_failure_if_the_servo_cannot_be_reactivated(make_node):
    """
    A successful home must not report success if the hand-back to the servo then fails.

    Otherwise the caller sees `homed: true` while the arm is left on fr3_arm_controller, unable to
    take a target chunk from the policy — the same silent-dead-policy failure the handover exists
    to avoid, just arriving one step later than the failure modes above cover.
    """
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)
    call_async = _stub_switch_sequence(node, [True, False])  # borrow succeeds, hand-back fails

    response = _home(node)

    assert not response.success
    assert 'failed to hand the arm back' in response.message
    assert _switch_pairs(call_async) == [
        ((mb.MOVEIT_CONTROLLER,), (mb.SERVO_CONTROLLER,)),
        ((mb.SERVO_CONTROLLER,), (mb.MOVEIT_CONTROLLER,)),
    ]


def test_home_without_the_servo_running_does_not_switch_back(make_node):
    """
    A declined handover must not be followed by a switch back.

    Running the MoveIt executor alone is the normal pre-Phase-4 configuration. There the first
    switch is declined (no such controller), and activating the servo afterwards would be wrong —
    it was never holding the arm.
    """
    node = make_node()
    _stub_plan_ok(node)
    _capture_execute(node)
    call_async = _stub_switch(node, ok=False)

    assert _home(node).success, 'a declined handover must not fail the home itself'

    assert _switch_pairs(call_async) == [((mb.MOVEIT_CONTROLLER,), (mb.SERVO_CONTROLLER,))]

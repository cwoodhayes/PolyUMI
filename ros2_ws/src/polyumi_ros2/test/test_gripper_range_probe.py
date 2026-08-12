"""
Tests for the FR3 gripper range probe.

The probe's output goes straight into a robot command, so the failures that matter are the quiet
ones: reporting an endpoint the hand never actually reached, or calling a non-repeatable stall
point a calibration. The settle logic is the sharp bit — it must not require motion, because
commanding open when the hand is already open legitimately moves nothing and the bridge's deadband
may swallow the goal entirely.
"""

from unittest.mock import MagicMock

import pytest
import rclpy
from rclpy.parameter import Parameter

from polyumi_ros2.gripper_range_probe import (
    REPEATABILITY_WARN_M,
    GripperRangeProbe,
)


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    """Init/shutdown rclpy once for the whole module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def _probe(**overrides):
    """Build a probe whose publisher records instead of publishing, with logging silenced."""
    params = [Parameter(k, value=v) for k, v in overrides.items()]
    node = GripperRangeProbe(parameter_overrides=params)
    sent: list = []
    node._pub = type(
        'P',
        (),
        {
            'publish': lambda _s, msg: sent.append(msg),
            'topic_name': '/polyumi/target_gripper',
            'get_subscription_count': lambda _s: 1,
        },
    )()
    node.get_logger = MagicMock()
    return node, sent


def test_command_publishes_a_single_waypoint_the_bridge_can_read():
    """The bridge reads points[0].positions[0]; anything else commands the wrong width."""
    node, sent = _probe()
    try:
        node.command(0.037)
    finally:
        node.destroy_node()

    assert len(sent) == 1
    assert list(sent[0].points[0].positions) == [0.037]
    assert list(sent[0].joint_names) == ['fr3_gripper_width']
    # A zero would make the bridge derive its maximum move speed, which slams the fingers shut.
    assert sent[0].points[0].time_from_start.sec == 1


def test_settle_returns_without_motion_when_the_hand_is_already_there():
    """
    Commanding open while already open moves nothing, and the deadband may drop the goal.

    A settle that waited for movement first would hang here for the full timeout — the arm-side
    lesson does not transfer, because the hand has no plan latency to wait out.
    """
    node, _ = _probe(settle_timeout_s=4.0)
    node._aperture = 0.0781
    try:
        assert node.settle() == pytest.approx(0.0781)
    finally:
        node.destroy_node()


def test_settle_gives_up_rather_than_hanging_when_nothing_is_reporting():
    """
    A hand that never reports must time out and return None, not block forever.

    None is the honest answer here — there is no last reading to fall back on — and _endpoint
    turns it into an error rather than a number that would land in a config file.
    """
    node, _ = _probe(settle_timeout_s=0.1)
    node._aperture = None
    try:
        assert node.settle() is None
    finally:
        node.destroy_node()


def test_report_accepts_repeatable_endpoints():
    """The happy path: tight spreads at both ends, exit 0."""
    node, _ = _probe(open_width_m=0.09)
    try:
        assert node._report([0.0782, 0.0781, 0.0782], [0.0061, 0.0061, 0.0062]) == 0
    finally:
        node.destroy_node()


def test_report_rejects_a_non_repeatable_closed_endpoint():
    """
    Move stalls on contact, so the closed endpoint can wander between reps.

    Calibrating against a wandering endpoint bakes that error into every commanded width, so this
    has to fail rather than average it away.
    """
    node, _ = _probe()
    closeds = [0.0060, 0.0060 + 3 * REPEATABILITY_WARN_M, 0.0061]
    try:
        assert node._report([0.0782, 0.0781, 0.0782], closeds) == 1
        message = str(node.get_logger().error.call_args_list)
    finally:
        node.destroy_node()

    assert 'use_grasp_below_m' in message, 'must name the fix, not just the symptom'


def test_landing_on_the_clamp_is_reported_as_a_software_limit():
    """
    Regression: the first hardware run reported open aperture = 0.0800 against a 0.08 clamp.

    That is the bridge refusing to command wider, not the fingers stopping — a measurement of the
    software, which reads exactly like a measurement of the hardware unless it is called out.
    """
    node, _ = _probe(open_width_m=0.09)
    try:
        assert node._report([0.08, 0.08, 0.08], [0.0, 0.0, 0.0], clamp_m=0.08) == 1
        errors = str(node.get_logger().error.call_args_list)
    finally:
        node.destroy_node()

    assert 'gripper_max_width:=0.0817' in errors, 'must name the exact re-run command'
    assert 'software limit' in errors


def test_sitting_at_a_clamp_that_is_the_hands_own_maximum_is_the_hardware_answer():
    """
    Regression: the second run reported 0.0816 against a 0.0817 clamp.

    Only 0.1 mm apart, so no margin separates "hit the clamp" from "hit the hardware" — and the
    spread was 0.00 mm in BOTH runs, so the earlier spread heuristic called this a clamp too. What
    settles it is that 0.0817 is the hand's own maximum: franka_gripper aborts anything wider, so
    there is nothing further to command and telling the operator to raise the clamp again would
    send them in a circle.
    """
    node, _ = _probe(open_width_m=0.09)
    try:
        assert node._report([0.0816, 0.0816, 0.0816], [0.0, 0.0, 0.0], clamp_m=0.0817) == 0
        info = str(node.get_logger().info.call_args_list)
    finally:
        node.destroy_node()

    assert 'nothing further to command' in info


def test_stopping_clear_of_the_clamp_is_reported_as_a_physical_stop():
    """The unambiguous case: the fingers foul well before the software would have stopped them."""
    node, _ = _probe(open_width_m=0.09)
    try:
        assert node._report([0.0700, 0.0701, 0.0700], [0.0, 0.0, 0.0], clamp_m=0.0817) == 0
        info = str(node.get_logger().info.call_args_list)
    finally:
        node.destroy_node()

    assert 'hardware stopped it' in info


def test_an_unreachable_bridge_param_is_admitted_not_guessed():
    """
    Without the clamp the verdict is unknowable, and saying so beats inventing one.

    Non-fatal on purpose: the endpoints were still measured, they are merely unverified.
    """
    node, _ = _probe(open_width_m=0.09)
    try:
        assert node._report([0.0816, 0.0816, 0.0816], [0.0, 0.0, 0.0], clamp_m=None) == 0
        warnings = str(node.get_logger().warning.call_args_list)
    finally:
        node.destroy_node()

    assert 'might be the' in warnings


@pytest.mark.parametrize(
    'bad',
    [
        {'reps': 0},
        {'open_width_m': 0.01, 'closed_width_m': 0.02},
    ],
)
def test_invalid_parameters_fail_fast(bad):
    """An inverted range would drive the hand backwards through the whole probe and 'pass'."""
    params = [Parameter(k, value=v) for k, v in bad.items()]
    with pytest.raises(ValueError):
        GripperRangeProbe(parameter_overrides=params)

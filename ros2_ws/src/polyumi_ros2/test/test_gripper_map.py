"""
Tests for the policy<->robot gripper width conversion.

The conversion is two lines of arithmetic, but it sits on a path where being wrong is invisible:
a bad offset just commands a systematically-too-open or too-closed hand, and a bad clamp just
saturates. Neither raises. These pin the direction of the conversion (the sign is the easy thing
to flip) and the clamping behaviour.
"""

import pytest

from polyumi_ros2.gripper_map import aperture_from_joint_state, policy_to_robot_width, robot_to_policy_width

#: A hypothetical set of fingers whose tips meet before the mechanism bottoms out. The real ones
#: measure 0.0 (see test_current_hardware_is_a_passthrough), which would make every test here pass
#: for the wrong reason, so the interesting cases use a non-zero value on purpose.
CLOSED_APERTURE = 0.006
MAX_WIDTH = 0.08


def test_policy_zero_is_the_closed_aperture():
    """Policy width 0 means 'fully closed', which on the arm is the closed aperture, not 0."""
    assert policy_to_robot_width(0.0, CLOSED_APERTURE, MAX_WIDTH) == pytest.approx(CLOSED_APERTURE)


def test_closed_aperture_reads_back_as_policy_zero():
    """The observation direction: a shut hand is 0 metres of opening, whatever its encoder says."""
    assert robot_to_policy_width(CLOSED_APERTURE, CLOSED_APERTURE) == pytest.approx(0.0)


def test_round_trip_within_range_is_identity():
    """Converting out and back changes nothing while the value stays in the robot's range."""
    for policy_width in (0.0, 0.005, 0.02, 0.05, 0.07):
        robot = policy_to_robot_width(policy_width, CLOSED_APERTURE, MAX_WIDTH)
        assert robot_to_policy_width(robot, CLOSED_APERTURE) == pytest.approx(policy_width)


def test_there_is_no_independent_offset_to_get_wrong():
    """
    The reason gripper_offset_m was removed: it was always exactly -min_width_m.

    Carrying both let one measurement sit behind two knobs of opposite sign, and setting only one
    was wrong by that amount across the whole range — while still reading correctly at the closed
    end, where the low clamp hid it. Deriving the offset from min_width_m makes that unpayable.
    """
    for closed_aperture in (0.0, 0.003, 0.006, 0.02):
        # The zero point always lands on the closed aperture; nothing else can be configured.
        assert policy_to_robot_width(0.0, closed_aperture, MAX_WIDTH) == pytest.approx(closed_aperture)
        # And an opening of d always lands d above it, for every d that stays in range.
        assert policy_to_robot_width(0.01, closed_aperture, MAX_WIDTH) == pytest.approx(closed_aperture + 0.01)


def test_slope_is_one():
    """
    The map is a pure offset, not a rescale.

    This mirrors UMI: in their repo, get_gripper_calibration_interpolator computes
    aruco_actual_width - aruco_min_width, and the dataset plan passes the same array as both of
    its arguments, so despite the interp1d machinery there is no gain term. A future change that
    introduces scaling should have to update this test deliberately.
    """
    delta = 0.01
    low = policy_to_robot_width(0.02, CLOSED_APERTURE, MAX_WIDTH)
    high = policy_to_robot_width(0.02 + delta, CLOSED_APERTURE, MAX_WIDTH)

    assert high - low == pytest.approx(delta)


def test_clamps_at_the_robot_maximum():
    """The handheld gripper opens wider than the hand, so over-wide commands saturate."""
    assert policy_to_robot_width(0.5, CLOSED_APERTURE, MAX_WIDTH) == pytest.approx(MAX_WIDTH)


def test_clamps_at_the_closed_aperture_not_at_zero():
    """
    Commanding below the closed aperture makes Move stall and abort.

    That is what leaves the hand node's deadband measuring against a width the hand never
    reached, so the floor is the fingers' real minimum rather than a hardcoded 0.
    """
    assert policy_to_robot_width(0.0, CLOSED_APERTURE, MAX_WIDTH) == pytest.approx(CLOSED_APERTURE)
    assert policy_to_robot_width(-1.0, CLOSED_APERTURE, MAX_WIDTH) == pytest.approx(CLOSED_APERTURE)


def test_observation_direction_is_not_clamped():
    """
    An out-of-range *measurement* passes through unclamped, on purpose.

    Clamping the observation would hide a miscalibrated closed aperture from both the policy and
    whoever is reading the logs; the robot's actual aperture is real information either way.
    """
    assert robot_to_policy_width(0.5, CLOSED_APERTURE) == pytest.approx(0.5 - CLOSED_APERTURE)
    assert robot_to_policy_width(0.0, CLOSED_APERTURE) == pytest.approx(-CLOSED_APERTURE)


def test_current_hardware_is_a_passthrough():
    """
    With the real fingers the closed aperture is 0.0, so the map is the identity plus clamping.

    Not a placeholder: these fingers meet at the mechanism's true zero, which puts us where UMI's
    WSG already is — their inference path converts nothing either.
    """
    assert policy_to_robot_width(0.03, 0.0, MAX_WIDTH) == pytest.approx(0.03)
    assert robot_to_policy_width(0.03, 0.0) == pytest.approx(0.03)


def _state(names, positions):
    """Build a JointState carrying these joint names and positions."""
    from sensor_msgs.msg import JointState

    msg = JointState()
    msg.name = list(names)
    msg.position = list(positions)
    return msg


def test_finger_joints_are_summed():
    """franka_hand_node follows franka_gripper: each finger joint carries half the aperture."""
    msg = _state(['fr3_finger_joint1', 'fr3_finger_joint2'], [0.0406, 0.0406])
    assert aperture_from_joint_state(msg) == pytest.approx(0.0812)


def test_width_joint_is_the_whole_aperture():
    """franka_gripper_control publishes one joint, fr3_gripper_width, already the full width."""
    msg = _state(['fr3_gripper_width'], [0.0812])
    assert aperture_from_joint_state(msg) == pytest.approx(0.0812)


def test_width_joint_is_found_by_name_not_by_position():
    """A driver may order or pad its joints however it likes; the name is what selects the width."""
    msg = _state(['some_other_joint', 'fr3_gripper_width'], [1.234, 0.0812])
    assert aperture_from_joint_state(msg) == pytest.approx(0.0812)


def test_unknown_joints_are_none_not_zero():
    """A message naming no width joint must not read as a closed gripper."""
    assert aperture_from_joint_state(_state([], [])) is None
    assert aperture_from_joint_state(_state(['elbow'], [0.5])) is None

"""
Tests for the policy<->robot gripper width conversion.

The conversion is two lines of arithmetic, but it sits on a path where being wrong is invisible:
a bad offset just commands a systematically-too-open or too-closed hand, and a bad clamp just
saturates. Neither raises. These pin the direction of the offset (the sign is the easy thing to
flip) and the clamping behaviour.
"""

import pytest

from polyumi_ros2.gripper_map import policy_to_robot_width, robot_to_policy_width

OFFSET = 0.005
MAX_WIDTH = 0.08


def test_offset_is_subtracted_going_to_the_robot():
    """A closed handheld gripper (tags still separated by the offset) maps to a closed hand."""
    assert policy_to_robot_width(OFFSET, OFFSET, MAX_WIDTH) == pytest.approx(0.0)


def test_offset_is_added_coming_from_the_robot():
    """A closed hand reads back as the offset, i.e. what the training data calls 'closed'."""
    assert robot_to_policy_width(0.0, OFFSET) == pytest.approx(OFFSET)


def test_round_trip_within_range_is_identity():
    """Converting out and back changes nothing while the value stays in the robot's range."""
    for policy_width in (0.005, 0.02, 0.05, 0.085):
        robot = policy_to_robot_width(policy_width, OFFSET, MAX_WIDTH)
        assert robot_to_policy_width(robot, OFFSET) == pytest.approx(policy_width)


def test_slope_is_one():
    """
    The map is a pure offset, not a rescale.

    This mirrors UMI: get_gripper_calibration_interpolator computes
    aruco_actual_width - aruco_min_width, and the dataset plan passes the same array as both of
    its arguments, so despite the interp1d machinery there is no gain term. A future change that
    introduces scaling should have to update this test deliberately.
    """
    delta = 0.01
    low = policy_to_robot_width(0.02, OFFSET, MAX_WIDTH)
    high = policy_to_robot_width(0.02 + delta, OFFSET, MAX_WIDTH)
    assert high - low == pytest.approx(delta)


def test_clamps_at_the_robot_maximum():
    """The handheld gripper opens wider than the hand, so over-wide commands saturate."""
    assert policy_to_robot_width(0.5, OFFSET, MAX_WIDTH) == pytest.approx(MAX_WIDTH)


def test_clamps_at_zero():
    """A width below the offset would imply a negative aperture; it floors at closed instead."""
    assert policy_to_robot_width(0.0, OFFSET, MAX_WIDTH) == pytest.approx(0.0)
    assert policy_to_robot_width(-1.0, OFFSET, MAX_WIDTH) == pytest.approx(0.0)


def test_observation_direction_is_not_clamped():
    """
    An out-of-range *measurement* passes through unclamped, on purpose.

    Clamping the observation would hide a miscalibrated offset from both the policy and whoever
    is reading the logs; the robot's actual aperture is real information either way.
    """
    assert robot_to_policy_width(0.5, OFFSET) == pytest.approx(0.505)


def test_zero_offset_is_a_passthrough():
    """With no offset configured the conversion degrades to a plain clamp."""
    assert policy_to_robot_width(0.03, 0.0, MAX_WIDTH) == pytest.approx(0.03)
    assert robot_to_policy_width(0.03, 0.0) == pytest.approx(0.03)

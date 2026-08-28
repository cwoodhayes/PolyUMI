"""
Check that `gripper` and `execute_gripper` select exactly one driver, or none.

The two grippers are mutually exclusive hardware and both claim their device on startup, so
"which one starts" is the one branch in this launch file that can do damage by being wrong.
Evaluating the conditions needs no ROS graph, no hardware, and no NUC.
"""

import importlib.util
import pathlib

import pytest
from launch import LaunchContext

_LAUNCH_FILE = pathlib.Path(__file__).resolve().parent / 'launch' / 'fr3_inference.launch.py'

#: Substrings that identify each driver's action in the LaunchDescription. The Franka Hand is a
#: Node with an executable; the FAULHABER comes in as an IncludeLaunchDescription, so it is
#: identified by the launch file it points at.
HAND = 'franka_hand_node'
FAULHABER = 'faulhaber_gripper.launch.xml'


def _drivers_started(gripper: str, execute_gripper: str) -> set:
    """
    Return which gripper drivers this argument combination would actually launch.

    :param gripper: value of the ``gripper`` launch argument.
    :param execute_gripper: value of the ``execute_gripper`` launch argument, as a string.
    :return: subset of ``{HAND, FAULHABER}``.
    """
    spec = importlib.util.spec_from_file_location('fr3_inference_launch', _LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    context = LaunchContext()
    context.launch_configurations['gripper'] = gripper
    context.launch_configurations['execute_gripper'] = execute_gripper

    started = set()
    for entity in module.generate_launch_description().entities:
        condition = getattr(entity, 'condition', None)
        if condition is None:
            continue
        # A Node names its executable; an include only names itself through the (unresolved,
        # but already stringified) location of the launch file it points at.
        source = getattr(entity, 'launch_description_source', None)
        text = str(getattr(entity, 'node_executable', '')) + str(getattr(source, 'location', ''))
        for name in (HAND, FAULHABER):
            if name in text and condition.evaluate(context):
                started.add(name)
    return started


@pytest.mark.parametrize(
    ('gripper', 'execute_gripper', 'expected'),
    [
        # The defaults. Nothing may move on a bare `ros2 launch`.
        ('hand', 'false', set()),
        ('faulhaber', 'false', set()),
        ('none', 'false', set()),
        # execute_gripper alone is not enough; `gripper` says which device.
        ('hand', 'true', {HAND}),
        ('faulhaber', 'true', {FAULHABER}),
        ('none', 'true', set()),
    ],
)
def test_exactly_one_driver_is_selected(gripper, execute_gripper, expected):
    """Never both drivers at once — they would fight over one set of fingers."""
    assert _drivers_started(gripper, execute_gripper) == expected

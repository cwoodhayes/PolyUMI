# PolyUMI: everything the NUC needs for inference ON TOP of a running hardware session.
#
# Collapses three terminals (move_group + the two bridges) into one. All three are plain ROS
# nodes with no hardware handshake, so they start together, fail together, and restart together
# without touching the arm's state.
#
# They are separate from fr3_bringup.launch.py on purpose — see the note at the top of that
# file. Start bringup first; this file's nodes all tolerate being started before their peers
# (move_group waits, the bridges log a "server NOT found" error and keep spinning), but the
# arm controller must already exist or move_group comes up with nothing to execute through.

"""
Launch the NUC-side inference stack: move_group plus the PolyUMI target bridges.

Run on the NUC, after fr3_bringup.launch.py is up:

    ros2 launch nuc/launch/fr3_inference.launch.py                       # dry run, nothing moves
    ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true # fingers only
    ros2 launch nuc/launch/fr3_inference.launch.py \
        execute_arm:=true execute_gripper:=true max_velocity_scaling:=0.2

See docs/crb-fr3-inference.md for the full bringup order and its gotchas.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# The bridges are standalone scripts, not an installed ament package (they run from a plain
# clone on the NUC, which has no PolyUMI workspace), so they are ExecuteProcess by path rather
# than Node by package name.
NUC_DIR = Path(__file__).resolve().parent.parent


def generate_launch_description():
    """Include move_group and start the gripper + MoveIt bridges."""
    robot_ip = LaunchConfiguration('robot_ip')
    execute_arm = LaunchConfiguration('execute_arm')
    execute_gripper = LaunchConfiguration('execute_gripper')
    max_velocity_scaling = LaunchConfiguration('max_velocity_scaling')
    gripper_max_width = LaunchConfiguration('gripper_max_width')

    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='192.168.51.20',
                              description='Hostname or IP of the FR3; forwarded to move_group.'),
        # Two execute flags, not one. docs/crb-fr3-inference.md recommends running the arm
        # plan-only while the gripper executes for a first hardware run, so a bad width moves
        # fingers and nothing else — a single shared flag would take that away. Both default
        # false: launching this file must never move the robot on its own.
        DeclareLaunchArgument('execute_arm', default_value='false',
                              description='Let fr3_moveit_bridge execute plans (MOVES THE ARM).'),
        DeclareLaunchArgument('execute_gripper', default_value='false',
                              description='Let fr3_gripper_bridge send goals (MOVES THE FINGERS).'),
        DeclareLaunchArgument('max_velocity_scaling', default_value='0.1',
                              description='Arm speed cap. Start low, raise once you trust it.'),
        DeclareLaunchArgument('gripper_max_width', default_value='0.0817',
                              description="Gripper aperture clamp (m). Raise to 0.0817 to measure "
                                          'the fingers true open limit with gripper_range_probe.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(NUC_DIR / 'launch' / 'fr3_move_group.launch.py')),
            launch_arguments={'robot_ip': robot_ip}.items(),
        ),

        ExecuteProcess(
            cmd=['python3', str(NUC_DIR / 'fr3_moveit_bridge.py'), '--ros-args',
                 '-p', ['execute:=', execute_arm],
                 '-p', ['max_velocity_scaling:=', max_velocity_scaling]],
            output='screen',
        ),
        ExecuteProcess(
            cmd=['python3', str(NUC_DIR / 'fr3_gripper_bridge.py'), '--ros-args',
                 '-p', ['execute:=', execute_gripper],
                 '-p', ['max_width_m:=', gripper_max_width]],
            output='screen',
        ),
    ])

# PolyUMI: the FR3 *hardware session*, as one launch file.
#
# Replaces the two-terminal `fr3-bringup` + `fr3-arm-controller` alias dance. Those were split
# only because the controller spawner has to run AFTER controller_manager exists — not because
# they are independent things. They are one unit: restarting bringup tears down
# controller_manager, which drops the controller, so the spawner has to run again anyway.
#
# What is deliberately NOT in here: move_group and the two PolyUMI bridges. Those live in
# fr3_inference.launch.py, so this file can be restarted on its own — which matters, because
# per docs/crb-fr3-inference.md ("TF lookup fails") this is the component that crashes
# mid-session, and it is also the one gated on enabling FCI in the Desk UI by hand.

"""
Launch the FR3 hardware session: franka_bringup plus the arm controller.

Run on the NUC, after enabling FCI on the Desk UI:

    ros2 launch nuc/launch/fr3_bringup.launch.py

See docs/crb-fr3-inference.md for the full bringup order and its gotchas.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# How long the spawner waits for controller_manager to appear. franka.launch.py brings the arm
# up inside an OpaqueFunction, so there is no node handle here to hang an OnProcessStart event
# on — upstream spawns its own broadcasters the same way, on the spawner's built-in wait. The
# default (10s) is tight when the robot is cold, and overshooting costs nothing: the spawner
# returns as soon as controller_manager answers.
CONTROLLER_MANAGER_TIMEOUT_S = '60'


def generate_launch_description():
    """Include franka_bringup and spawn fr3_arm_controller once controller_manager is up."""
    robot_ip = LaunchConfiguration('robot_ip')
    arm_id = LaunchConfiguration('arm_id')
    load_gripper = LaunchConfiguration('load_gripper')

    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='192.168.51.20',
                              description='Hostname or IP of the FR3 (the NUC-side .51 link).'),
        DeclareLaunchArgument('arm_id', default_value='fr3',
                              description='Arm type; drives every fr3_* frame and topic name.'),
        # franka.launch.py defaults this to true as well. Named here because turning it off is
        # what makes the /fr3_gripper/* action servers vanish — the first thing to check when
        # the gripper bridge reports "action server NOT found".
        DeclareLaunchArgument('load_gripper', default_value='true',
                              description='Load the Franka Hand (the /fr3_gripper/* servers).'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])]),
            launch_arguments={
                'robot_ip': robot_ip,
                'arm_id': arm_id,
                'load_gripper': load_gripper,
            }.items(),
        ),

        # The joint-trajectory controller move_group executes through. franka.launch.py spawns
        # only the two broadcasters, so without this /execute_trajectory has nothing to drive
        # and the arm never moves. --param-file declares the controller and its gains; -t is
        # redundant alongside it but is what the known-good `fr3-arm-controller` alias passed,
        # and this file only ever runs on the Humble NUC, whose spawner still accepts it.
        Node(
            package='controller_manager',
            executable='spawner',
            name='fr3_arm_controller_spawner',
            output='screen',
            arguments=[
                'fr3_arm_controller',
                '-t', 'joint_trajectory_controller/JointTrajectoryController',
                '--param-file', PathJoinSubstitution([
                    FindPackageShare('franka_fr3_moveit_config'),
                    'config', 'fr3_ros_controllers.yaml',
                ]),
                '--controller-manager-timeout', CONTROLLER_MANAGER_TIMEOUT_S,
            ],
        ),
    ])

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

from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare

# ros2 launch loads this file by path without touching sys.path, so a sibling import needs the
# repo's nuc/ directory put there by hand. Same NUC_DIR idiom as fr3_inference.launch.py.
NUC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NUC_DIR))

import tcp_calib  # noqa: E402

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

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'robot_ip',
                default_value='192.168.51.20',
                description='Hostname or IP of the FR3 (the NUC-side .51 link).',
            ),
            DeclareLaunchArgument(
                'arm_id', default_value='fr3', description='Arm type; drives every fr3_* frame and topic name.'
            ),
            # NOT "does the hand exist" — it means "franka_gripper owns the hand instead of us".
            # PolyUMI's franka_hand_node talks to libfranka directly and only one process can hold
            # that connection, so the default is false. Set it true to hand the fingers back to the
            # stock /fr3_gripper/* action servers, and then do NOT launch franka_hand_node.
            #
            # Beware the side effect: franka.launch.py routes this one flag into `xacro hand:=` as
            # well, so turning it off also drops fr3_hand from robot_description. The static
            # publisher below is what keeps polyumi_tcp attached to something.
            DeclareLaunchArgument(
                'load_gripper',
                default_value='false',
                description='Let franka_gripper own the Franka Hand instead of franka_hand_node.',
            ),
            # franka.launch.py:147 hardcodes joint_state_publisher's
            # `source_list: [franka/joint_states, franka_gripper/joint_states]`, but the gripper —
            # ours or franka_gripper — publishes on `fr3_gripper/joint_states`, arm_id-prefixed. So
            # the aggregator subscribes to a topic nothing publishes, and the fingers never reach
            # /joint_states. Upstream franka_ros2 v0.1.15 bug.
            #
            # A remap is the only lever that reaches it: SetParameter loses, because launch_ros
            # expands global parameters FIRST precisely so a node's own hardcoded ones win
            # (launch_ros/actions/node.py:422), while global REMAPS are prepended (node.py:471) and
            # joint_state_publisher declares none of its own. Scoped so it touches nothing else.
            #
            # This only pays off under load_gripper:=true. On the default path the URDF has no hand
            # at all, so joint_state_publisher drops fr3_finger_joint1/2 as joints it does not know
            # (joint_state_publisher.py:332) no matter which topic they arrive on — and move_group,
            # whose own model DOES have the fingers, keeps warning "complete state ... not yet
            # known". Fixing that needs a hand in robot_description, which is a different problem.
            GroupAction(
                scoped=True,
                actions=[
                    SetRemap('/franka_gripper/joint_states', '/fr3_gripper/joint_states'),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            [PathJoinSubstitution([FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])]
                        ),
                        launch_arguments={
                            'robot_ip': robot_ip,
                            'arm_id': arm_id,
                            'load_gripper': load_gripper,
                        }.items(),
                    ),
                ],
            ),
            # The joint-trajectory controller move_group executes through. franka.launch.py spawns
            # only the two broadcasters, so without this /execute_trajectory has nothing to drive
            # and the arm never moves. The two arguments do DIFFERENT jobs and both are required:
            # Humble's spawner sets the `type` param only from -t (spawner.py:208), while
            # --param-file goes through set_controller_parameters_from_param_files, which sets the
            # controller's `params_file` and never reads a type out of it. Dropping -t fails with
            # "The 'type' param was not defined for ...", pointing at the controller rather than at
            # the missing flag.
            Node(
                package='controller_manager',
                executable='spawner',
                name='fr3_arm_controller_spawner',
                output='screen',
                arguments=[
                    'fr3_arm_controller',
                    '-t',
                    'joint_trajectory_controller/JointTrajectoryController',
                    '--param-file',
                    PathJoinSubstitution(
                        [
                            FindPackageShare('franka_fr3_moveit_config'),
                            'config',
                            'fr3_ros_controllers.yaml',
                        ]
                    ),
                    '--controller-manager-timeout',
                    CONTROLLER_MANAGER_TIMEOUT_S,
                ],
            ),
            # The frame the policy actually speaks in. It lives here rather than in the inference
            # launch so it exists whenever the arm does — Foxglove and `tf2_echo fr3_hand
            # polyumi_tcp` are how you check the calibration, and neither should need move_group.
            # franka.launch.py owns robot_state_publisher and hardcodes its xacro mappings, so an
            # extra URDF link cannot be threaded through it; a static publisher is the way in.
            # move_group gets the same numbers as xacro args — see nuc/tcp_calib.py.
            LogInfo(msg=f'[fr3_bringup] TCP {tcp_calib.describe()}'),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='polyumi_tcp_static_tf',
                output='screen',
                arguments=tcp_calib.static_transform_publisher_args(),
            ),
            # With load_gripper false the URDF has no fr3_hand, so robot_state_publisher stops
            # emitting fr3_link8 -> fr3_hand and polyumi_tcp above becomes an orphan. That breaks
            # the laptop's base -> polyumi_tcp lookup, and with it every observation — a failure
            # whose symptom points nowhere near this flag. Republish the joint the URDF lost.
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='fr3_hand_static_tf',
                output='screen',
                arguments=tcp_calib.hand_transform_publisher_args(),
                condition=UnlessCondition(load_gripper),
            ),
        ]
    )

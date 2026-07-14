#  Copyright (c) 2024 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# PolyUMI note: this is franka_fr3_moveit_config/launch/move_group.launch.py, fixed and
# enriched to be a functional *standalone* move_group that runs ALONGSIDE an already-up
# `fr3-bringup` on the NUC — it starts ONLY the move_group node (no hardware, no
# controllers, no robot_state_publisher), so there is no collision.
#
# Two changes vs. upstream:
#   1. Declare robot_ip / use_fake_hardware / fake_sensor_commands. Upstream references
#      these LaunchConfigurations without declaring them, so it can't launch at all
#      ("launch configuration 'fake_sensor_commands' does not exist").
#   2. Pass the move_group params that upstream omits (OMPL pipeline, trajectory
#      execution, the simple controller manager -> fr3_arm_controller, and the planning
#      scene monitor). Without these, upstream's bare move_group defaults to CHOMP and
#      logs "No controller_names specified" -> /execute_trajectory cannot move the arm.
#      These are copied from franka_fr3_moveit_config's OWN moveit.launch.py (minus the
#      hardware/RViz/controller nodes that fr3-bringup already provides).
# See docs/crb-fr3-inference.md for how to run this and the gotchas around it.

"""Launch a standalone MoveIt move_group for the FR3, alongside a running fr3-bringup."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import yaml


def load_yaml(package_name, file_path):
    """Load a YAML config file from a package's share directory, or None if unreadable."""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* Windows Error where available
        return None


def generate_launch_description():
    """Build the launch description: declare args and start the move_group node."""
    robot_ip_parameter_name = 'robot_ip'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)

    # --- PolyUMI fix: declare the args the upstream file forgot to declare ---
    robot_ip_arg = DeclareLaunchArgument(
        robot_ip_parameter_name,
        default_value='192.168.51.20',
        description='Hostname or IP address of the FR3 robot.',
    )
    use_fake_hardware_arg = DeclareLaunchArgument(
        use_fake_hardware_parameter_name,
        default_value='false',
        description='Use fake (mock) hardware instead of the real robot.',
    )
    fake_sensor_commands_arg = DeclareLaunchArgument(
        fake_sensor_commands_parameter_name,
        default_value='false',
        description="Fake sensor commands. Only valid when 'use_fake_hardware' is true.",
    )

    db_arg = DeclareLaunchArgument('db', default_value='False', description='Database flag')

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'), 'robots', 'fr3', 'fr3.urdf.xacro'
    )

    robot_description_command = Command(
        [
            FindExecutable(name='xacro'),
            ' ',
            franka_xacro_file,
            ' ros2_control:=false',
            ' hand:=true',
            ' arm_id:=fr3',
            ' robot_ip:=',
            robot_ip,
            ' use_fake_hardware:=',
            use_fake_hardware,
            ' fake_sensor_commands:=',
            fake_sensor_commands,
        ]
    )

    robot_description = {'robot_description': ParameterValue(robot_description_command, value_type=str)}

    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'), 'srdf', 'fr3_arm.srdf.xacro'
    )

    robot_description_semantic_command = Command(
        [FindExecutable(name='xacro'), ' ', franka_semantic_xacro_file, ' hand:=true']
    )

    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(robot_description_semantic_command, value_type=str)
    }

    kinematics_yaml = load_yaml('franka_fr3_moveit_config', 'config/kinematics.yaml')

    # --- PolyUMI: the move_group params upstream move_group.launch.py omits ---
    # OMPL planning pipeline (upstream defaults to CHOMP without this).
    ompl_planning_pipeline_config = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
            'default_planner_request_adapters/ResolveConstraintFrames '
            'default_planner_request_adapters/FixWorkspaceBounds '
            'default_planner_request_adapters/FixStartStateBounds '
            'default_planner_request_adapters/FixStartStateCollision '
            'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl_planning_yaml = load_yaml('franka_fr3_moveit_config', 'config/ompl_planning.yaml')
    ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)

    # Trajectory execution: map MoveIt to the fr3_arm_controller that fr3-bringup runs
    # (fixes "No controller_names specified" -> /execute_trajectory can actually move).
    moveit_simple_controllers_yaml = load_yaml('franka_fr3_moveit_config', 'config/fr3_controllers.yaml')
    moveit_controllers = {
        'moveit_simple_controller_manager': moveit_simple_controllers_yaml,
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }
    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }

    # Planning scene monitor: subscribe to the live /joint_states from bringup so plans
    # start from the REAL robot state (fixes Cartesian fraction=0.0).
    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    run_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
        ],
    )

    return LaunchDescription(
        [
            robot_ip_arg,
            use_fake_hardware_arg,
            fake_sensor_commands_arg,
            db_arg,
            run_move_group_node,
        ]
    )

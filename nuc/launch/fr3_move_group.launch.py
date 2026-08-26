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
# Changes vs. upstream:
#   1. Declare robot_ip / use_fake_hardware / fake_sensor_commands. Upstream references
#      these LaunchConfigurations without declaring them, so it can't launch at all
#      ("launch configuration 'fake_sensor_commands' does not exist").
#   2. Pass the move_group params that upstream omits (OMPL pipeline, trajectory
#      execution, the simple controller manager -> fr3_arm_controller, and the planning
#      scene monitor). Without these, upstream's bare move_group defaults to CHOMP and
#      logs "No controller_names specified" -> /execute_trajectory cannot move the arm.
#      These are copied from franka_fr3_moveit_config's OWN moveit.launch.py (minus the
#      hardware/RViz/controller nodes that fr3-bringup already provides).
#   3. Build robot_description from nuc/description/fr3_polyumi.urdf.xacro instead of
#      franka_description's fr3.urdf.xacro, so the model carries `polyumi_tcp`.
#   4. Supply joint ACCELERATION limits. Neither upstream nor franka_description declares any,
#      so move_group's time parameterization falls back to 1 rad/s^2 and every Cartesian chunk
#      comes back slower than the policy asked for — the bridge then never hits the commanded
#      timeline and drops most chunks as still-busy.
#   5. Silence planning_scene_monitor's finger-joint warning. This move_group's OWN model
#      (fr3_polyumi.urdf.xacro) carries the hand, but fr3_bringup's robot_description does
#      not (franka.launch.py couples hand: to load_gripper:, and that's false so
#      franka_hand_node can own the connection instead of franka_gripper) -- so
#      /joint_states, sourced from THAT model, never reports fr3_finger_joint1/2 and this
#      monitor repeats "not yet known" at ~1 Hz forever. Harmless: /polyumi/home only plans the
#      fr3_arm group, which doesn't include the fingers.
#      Fixing it for real means decoupling hand: from load_gripper: in fr3_bringup's own
#      robot_description, which is a bigger, riskier change than a log line justifies.
# See docs/crb-fr3-inference.md for how to run this and the gotchas around it.

"""Launch a standalone MoveIt move_group for the FR3, alongside a running fr3-bringup."""

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import yaml

# ros2 launch loads this file by path without touching sys.path, so a sibling import needs the
# repo's nuc/ directory put there by hand. Same NUC_DIR idiom as fr3_inference.launch.py.
NUC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NUC_DIR))

import tcp_calib  # noqa: E402

#: The fr3_arm planning group's joints — the ones whose timing the Cartesian chunks depend on.
ARM_JOINT_NAMES = [f'fr3_joint{i}' for i in range(1, 8)]


def load_yaml(package_name, file_path):
    """Load a YAML config file from a package's share directory, or None if unreadable."""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* Windows Error where available
        return None


def load_required_yaml(package_name, file_path):
    """Load a YAML config file, raising a clear error instead of silently passing None downstream."""
    data = load_yaml(package_name, file_path)
    if data is None:
        raise RuntimeError(
            f"Could not load required config '{file_path}' from package '{package_name}' — "
            'file missing, unreadable, or empty. Is franka_fr3_moveit_config installed and sourced?'
        )
    return data


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

    # See header change 4. The ceiling move_group time-parameterizes a homing plan against.
    # Deliberately NOT the FR3's datasheet maximum: conservative, raise it while watching — the
    # arm will fire a reflex if a sweep is too aggressive. The bridge's max_velocity_scaling
    # then stretches that plan back out, so this sets what "scale 1.0" means rather than the
    # speed the arm actually runs at.
    max_acceleration_arg = DeclareLaunchArgument(
        'max_acceleration',
        default_value='1.5',
        description='Joint acceleration limit (rad/s^2) for trajectory time parameterization.',
    )

    # PolyUMI change 3 (see header): the stock fr3.urdf.xacro, wrapped so move_group's
    # RobotModel also carries `polyumi_tcp` — the frame the bridge names as GetCartesianPath's
    # link_name. TF gets the same transform from fr3_bringup.launch.py; both read tcp_calib.
    franka_xacro_file = str(NUC_DIR / 'description' / 'fr3_polyumi.urdf.xacro')

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
            *tcp_calib.xacro_args(),
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

    kinematics_yaml = load_required_yaml('franka_fr3_moveit_config', 'config/kinematics.yaml')

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
    ompl_planning_yaml = load_required_yaml('franka_fr3_moveit_config', 'config/ompl_planning.yaml')
    ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)

    # Trajectory execution: map MoveIt to the fr3_arm_controller that fr3-bringup runs
    # (fixes "No controller_names specified" -> /execute_trajectory can actually move).
    moveit_simple_controllers_yaml = load_required_yaml('franka_fr3_moveit_config', 'config/fr3_controllers.yaml')
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

    # value_type=float matters: a launch argument arrives as a string, and MoveIt would reject
    # (or silently ignore) a string where it wants a double.
    max_acceleration = ParameterValue(LaunchConfiguration('max_acceleration'), value_type=float)
    joint_limits = {
        'robot_description_planning': {
            'joint_limits': {
                # Acceleration only. Velocity limits already reach the RobotModel from the URDF
                # (franka_description/robots/fr3/joint_limits.yaml, 2.62-5.26 rad/s) and were
                # never the binding constraint; restating them here would be a second copy to
                # drift.
                name: {'has_acceleration_limits': True, 'max_acceleration': max_acceleration}
                for name in ARM_JOINT_NAMES
            }
        }
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
            joint_limits,
        ],
        # See change 5 above: fr3_bringup's robot_description never reports the finger joints
        # this move_group's own model expects, so this fires at ~1 Hz forever and is pure noise.
        # Scoped to the one logger so a real planning_scene_monitor problem still surfaces.
        arguments=[
            '--ros-args',
            '--log-level',
            'moveit_ros.planning_scene_monitor.planning_scene_monitor:=error',
        ],
    )

    return LaunchDescription(
        [
            robot_ip_arg,
            use_fake_hardware_arg,
            fake_sensor_commands_arg,
            db_arg,
            max_acceleration_arg,
            run_move_group_node,
        ]
    )

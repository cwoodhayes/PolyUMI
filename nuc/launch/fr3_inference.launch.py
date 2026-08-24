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
    ros2 launch nuc/launch/fr3_inference.launch.py execute_arm:=true     # servo drives the arm
    ros2 launch nuc/launch/fr3_inference.launch.py \
        executor:=moveit execute_arm:=true max_velocity_scaling:=0.2     # the legacy path

`executor` (default `servo`) decides which controller holds the arm: the streaming impedance
controller, or fr3_arm_controller for move_group. With `executor:=servo` the impedance controller is
only ACTIVATED when `execute_arm:=true`; otherwise it is loaded inactive and nothing moves.

It must MATCH the laptop's `wire` parameter on policy_client_node (`multidof` for servo,
`pose_array` for moveit), which is what decides where the chunks are sent. A mismatch is loud, not
silent: nothing subscribes to what the client publishes, so the arm does not move and the client
warns every second naming the topic and what should be listening.

See docs/crb-fr3-inference.md for the full bringup order and its gotchas.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SERVO_CONTROLLER = 'polyumi_cartesian_impedance_controller'
MOVEIT_CONTROLLER = 'fr3_arm_controller'

# fr3_moveit_bridge is a standalone script, not an installed ament package (it runs from a plain
# clone on the NUC, which has no PolyUMI workspace), so it is ExecuteProcess by path rather than
# Node by package name. franka_hand_node is C++ and does come from a built package.
NUC_DIR = Path(__file__).resolve().parent.parent


def generate_launch_description():
    """Include move_group and start the hand driver + the MoveIt bridge."""
    robot_ip = LaunchConfiguration('robot_ip')
    execute_arm = LaunchConfiguration('execute_arm')
    execute_gripper = LaunchConfiguration('execute_gripper')
    max_velocity_scaling = LaunchConfiguration('max_velocity_scaling')
    max_acceleration = LaunchConfiguration('max_acceleration')
    gripper_max_width = LaunchConfiguration('gripper_max_width')
    executor = LaunchConfiguration('executor')

    # Torque control starts the moment the controller activates, so it is gated on the same flag
    # as every other way this file can move the arm.
    activate_servo = PythonExpression(["'", executor, "' == 'servo' and '", execute_arm, "' == 'true'"])

    # Hoisted out of the LaunchDescription list so the event handler below can name it: the switch
    # has to wait for this to finish, and launch offers no ordering guarantee otherwise.
    impedance_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name='polyumi_impedance_controller_spawner',
        output='screen',
        arguments=[
            SERVO_CONTROLLER,
            # Required, and not redundant with --param-file: Humble's spawner sets the `type`
            # param only from -t. See the note in fr3_bringup.launch.py.
            '-t',
            'polyumi_fr3_controllers/CartesianImpedanceController',
            '--param-file',
            str(NUC_DIR / 'config' / 'polyumi_controllers.yaml'),
            # Inactive in both modes. Activating means claiming the effort interfaces
            # fr3_arm_controller already holds, which a spawner cannot do — that takes the switch.
            '--inactive',
            '--controller-manager-timeout',
            '30',
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'executor',
                default_value='servo',
                # Constrained, because activate_servo falls through to its 'not servo' branch on
                # any unrecognised value: a typo would load the impedance controller and never
                # activate it, leaving a stack where nothing drives the arm and nothing says why.
                choices=['servo', 'moveit'],
                description="Which controller holds the arm: 'servo' (the streaming Cartesian "
                'impedance controller) or "moveit" (fr3_arm_controller, plan-then-execute, the '
                "path it replaces). Must match policy_client_node's `wire` on the laptop. With "
                'executor:=servo the controller is only ACTIVATED if execute_arm is also true; '
                'otherwise it is loaded inactive and nothing moves.',
            ),
            DeclareLaunchArgument(
                'robot_ip',
                default_value='192.168.51.20',
                description='Hostname or IP of the FR3; forwarded to move_group.',
            ),
            # Two execute flags, not one. docs/crb-fr3-inference.md recommends running the arm
            # plan-only while the gripper executes for a first hardware run, so a bad width moves
            # fingers and nothing else — a single shared flag would take that away. Both default
            # false: launching this file must never move the robot on its own.
            DeclareLaunchArgument(
                'execute_arm', default_value='false', description='Let fr3_moveit_bridge execute plans (MOVES THE ARM).'
            ),
            DeclareLaunchArgument(
                'execute_gripper',
                default_value='false',
                description='Let franka_hand_node issue Moves (MOVES THE FINGERS). False still '
                'plans and logs every command at the real cadence.',
            ),
            DeclareLaunchArgument(
                'max_velocity_scaling',
                default_value='0.1',
                description='Arm speed cap. Start low, raise once you trust it.',
            ),
            DeclareLaunchArgument(
                'max_acceleration',
                default_value='1.5',
                description='Joint acceleration limit (rad/s^2) move_group time-parameterizes '
                'against. Forwarded to fr3_move_group.launch.py; without it MoveIt '
                'defaults to 1 rad/s^2 and plans chunks slower than the policy asked '
                'for. Distinct from max_velocity_scaling, which caps the RESULT.',
            ),
            DeclareLaunchArgument(
                'gripper_max_width',
                default_value='0.0',
                description='Node-side aperture clamp (m). 0.0 means ask the hand for its own '
                'max_width, which is the right answer and needs no constant here; '
                'gripper_range_probe needs the full stroke to measure open aperture. '
                'policy_client_node clamps to its own (measured) gripper_max_width_m '
                'first, so this is only a backstop.',
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(NUC_DIR / 'launch' / 'fr3_move_group.launch.py')),
                launch_arguments={'robot_ip': robot_ip, 'max_acceleration': max_acceleration}.items(),
            ),
            # Always started: it owns /polyumi/home, which both executors need, and homing borrows
            # the arm back from the servo. Its chunk subscription simply stays idle under
            # executor:=servo, because the client is then publishing the other wire format.
            ExecuteProcess(
                cmd=[
                    'python3',
                    str(NUC_DIR / 'fr3_moveit_bridge.py'),
                    '--ros-args',
                    '-p',
                    ['execute:=', execute_arm],
                    '-p',
                    ['max_velocity_scaling:=', max_velocity_scaling],
                ],
                output='screen',
            ),
            impedance_spawner,
            # Hand the arm to the servo, once the spawner has actually loaded it. Ordered on the
            # spawner's exit rather than declared alongside it because launch gives no ordering
            # guarantee, and switching to a controller that is not loaded yet just fails.
            RegisterEventHandler(
                OnProcessExit(
                    target_action=impedance_spawner,
                    on_exit=[
                        ExecuteProcess(
                            cmd=[
                                'ros2',
                                'control',
                                'switch_controllers',
                                '--deactivate',
                                MOVEIT_CONTROLLER,
                                '--activate',
                                SERVO_CONTROLLER,
                            ],
                            output='screen',
                            condition=IfCondition(activate_servo),
                        )
                    ],
                )
            ),
            # Owns the libfranka gripper connection outright, so fr3_bringup must run with
            # load_gripper:=false (its default) or the two fight over the hand. Named fr3_gripper
            # so ~/joint_states resolves to /fr3_gripper/joint_states, exactly as franka_gripper's
            # did — every existing consumer of that topic keeps working unchanged.
            Node(
                package='polyumi_fr3_controllers',
                executable='franka_hand_node',
                name='fr3_gripper',
                output='screen',
                parameters=[
                    {
                        # value_type is not optional: a LaunchConfiguration resolves to a STRING,
                        # and declare_parameter's bool/double defaults reject one outright.
                        'execute': ParameterValue(execute_gripper, value_type=bool),
                        'robot_ip': ParameterValue(robot_ip, value_type=str),
                        'max_width_m': ParameterValue(gripper_max_width, value_type=float),
                    }
                ],
            ),
        ]
    )

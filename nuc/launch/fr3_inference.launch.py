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

SERVO_CONTROLLER = 'polyumi_cartesian_impedance_controller'
MOVEIT_CONTROLLER = 'fr3_arm_controller'

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
                description='Let fr3_gripper_bridge send goals (MOVES THE FINGERS).',
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
                default_value='0.0817',
                description='Bridge-side aperture clamp (m). Defaults to the Franka '
                "Hand's own maximum so nothing software-side narrows the "
                'range: franka_gripper aborts wider goals anyway, and '
                'gripper_range_probe needs the full stroke to measure '
                'open aperture. policy_client_node clamps to its own (measured) '
                'gripper_max_width_m first, so this is a backstop.',
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
            ExecuteProcess(
                cmd=[
                    'python3',
                    str(NUC_DIR / 'fr3_gripper_bridge.py'),
                    '--ros-args',
                    '-p',
                    ['execute:=', execute_gripper],
                    '-p',
                    ['max_width_m:=', gripper_max_width],
                ],
                output='screen',
            ),
        ]
    )

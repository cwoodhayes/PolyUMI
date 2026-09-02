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

`gripper:=faulhaber|hand|none` picks the driver (faulhaber is the supported one);
`execute_gripper` decides whether it is started at all.

The streaming Cartesian impedance controller is loaded inactive and only ACTIVATED once
`execute_arm:=true`; otherwise it sits loaded but idle and nothing moves.

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
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

SERVO_CONTROLLER = 'polyumi_cartesian_impedance_controller'
MOVEIT_CONTROLLER = 'fr3_arm_controller'

# fr3_home_service is a standalone script, not an installed ament package (it runs from a plain
# clone on the NUC, which has no PolyUMI workspace), so it is ExecuteProcess by path rather than
# Node by package name. franka_hand_node is C++ and does come from a built package.
NUC_DIR = Path(__file__).resolve().parent.parent


def generate_launch_description():
    """Include move_group and start the hand driver + the MoveIt bridge."""
    robot_ip = LaunchConfiguration('robot_ip')
    execute_arm = LaunchConfiguration('execute_arm')
    execute_gripper = LaunchConfiguration('execute_gripper')
    gripper = LaunchConfiguration('gripper')
    max_acceleration = LaunchConfiguration('max_acceleration')

    # Torque control starts the moment the controller activates, so it is gated on the same flag
    # as every other way this file can move the arm.
    activate_servo = PythonExpression(["'", execute_arm, "' == 'true'"])

    # Which hardware, and whether it is allowed to move, stay two separate questions: the
    # first-run pattern in docs/crb-fr3-inference.md is a live gripper against a plan-only arm.
    def _selected(name: str) -> PythonExpression:
        return PythonExpression(["'", gripper, "' == '", name, "' and '", execute_gripper, "' == 'true'"])

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
            'franka_streaming_impedance_controller/CartesianImpedanceController',
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
                'robot_ip',
                default_value='192.168.51.20',
                description='Hostname or IP of the FR3; forwarded to move_group.',
            ),
            # Two execute flags, not one. docs/crb-fr3-inference.md recommends running the arm
            # plan-only while the gripper executes for a first hardware run, so a bad width moves
            # fingers and nothing else — a single shared flag would take that away. Both default
            # false: launching this file must never move the robot on its own.
            DeclareLaunchArgument(
                'execute_arm',
                default_value='false',
                description='Activate the streaming Cartesian impedance controller (MOVES THE ARM).',
            ),
            DeclareLaunchArgument(
                'execute_gripper',
                default_value='false',
                description='Start the driver named by `gripper` and let it move the fingers '
                '(MOVES THE FINGERS). False does not start it at all: both drivers claim their '
                'hardware on startup, so there is no dry run — use `gripper:=none` for an '
                'arm-only session.',
            ),
            # The two grippers are mutually exclusive hardware, so this selects one rather than
            # gating each.
            DeclareLaunchArgument(
                'gripper',
                default_value='faulhaber',
                choices=['hand', 'faulhaber', 'none'],
                description='Which gripper driver to run: `faulhaber` = franka_gripper_control '
                '(FAULHABER over CANopen, needs can0 up and a completed '
                '/faulhaber_gripper/calibrate) — the supported one; `hand` = franka_hand_node (a '
                'stock Franka Hand over libfranka, kept working but not what we run); `none` = '
                'neither, for an arm-only session. Only takes effect with execute_gripper:=true.',
            ),
            DeclareLaunchArgument(
                'max_acceleration',
                default_value='1.5',
                description='Joint acceleration limit (rad/s^2) move_group time-parameterizes '
                'against, for homing. Forwarded to fr3_move_group.launch.py; without it '
                'MoveIt defaults to 1 rad/s^2 and a home sweep crawls.',
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(NUC_DIR / 'launch' / 'fr3_move_group.launch.py')),
                launch_arguments={'robot_ip': robot_ip, 'max_acceleration': max_acceleration}.items(),
            ),
            # Always started: it owns /polyumi/home, and homing borrows the arm back from the
            # servo.
            ExecuteProcess(
                cmd=['python3', str(NUC_DIR / 'fr3_home_service.py')],
                output='screen',
            ),
            impedance_spawner,
            # Hand the arm to the servo, once the spawner has actually loaded it. Ordered on the
            # spawner's exit rather than declared alongside it because launch gives no ordering
            # guarantee, and switching to a controller that is not loaded yet just fails.
            #
            # `ros2 service call`, NOT `ros2 control switch_controllers`, and the difference is not
            # cosmetic. ros2controlcli goes through NodeStrategy (switch_controllers.py:80), i.e.
            # the ros2cli daemon — and one poisoned daemon takes this down with
            # `RuntimeError: !rclpy.ok()` while every other pane looks healthy. The arm then simply
            # never moves, because fr3_arm_controller still holds it and the servo stays inactive.
            # `ros2 service call` calls rclpy.init() and creates its own node (call.py:82), so it
            # has no daemon to be poisoned by. If the daemon is wedged, `ros2 daemon stop`.
            RegisterEventHandler(
                OnProcessExit(
                    target_action=impedance_spawner,
                    on_exit=[
                        ExecuteProcess(
                            cmd=[
                                'ros2',
                                'service',
                                'call',
                                '/controller_manager/switch_controller',
                                'controller_manager_msgs/srv/SwitchController',
                                # strictness 2 = STRICT: fail loudly rather than half-switch and
                                # leave two controllers claiming the same effort interfaces.
                                f"{{activate_controllers: ['{SERVO_CONTROLLER}'], "
                                f"deactivate_controllers: ['{MOVEIT_CONTROLLER}'], strictness: 2}}",
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
                package='franka_streaming_impedance_controller',
                executable='franka_hand_node',
                name='fr3_gripper',
                output='screen',
                condition=IfCondition(_selected('hand')),
                # value_type is not optional: a LaunchConfiguration resolves to a STRING, and
                # declare_parameter's str default would otherwise be the only one that fits.
                #
                # target_topic is explicit because the node's own default is node-relative
                # (~/target_widths, i.e. under the name below); the FAULHABER driver subscribes
                # /polyumi/target_gripper, and the two drivers must be interchangeable.
                parameters=[
                    {
                        'robot_ip': ParameterValue(robot_ip, value_type=str),
                        'target_topic': '/polyumi/target_gripper',
                    }
                ],
            ),
            # The FAULHABER gripper, from the franka_gripper_control submodule. It already
            # subscribes /polyumi/target_gripper as a JointTrajectory carrying fr3_gripper_width
            # in metres — the same contract franka_hand_node serves — so the ONLY thing this
            # override does is move its state stream onto the topic the six existing consumers
            # read. Everything else is left at its defaults; retune via its own launch args (see
            # its README), never `--ros-args -p`: its knobs are argparse, not ROS parameters.
            #
            # It holds position until /faulhaber_gripper/calibrate has found both hard stops, so a
            # fresh NUC needs that service call before any width is tracked.
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare('franka_gripper_control'),
                            'launch',
                            'faulhaber_gripper.launch.xml',
                        ]
                    )
                ),
                launch_arguments={
                    'joint_state_topic': '/fr3_gripper/joint_states',
                }.items(),
                condition=IfCondition(_selected('faulhaber')),
            ),
        ]
    )

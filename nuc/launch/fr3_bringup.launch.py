# PolyUMI: the FR3 *hardware session*, as one launch file.
#
# Replaces the two-terminal `fr3-bringup` + `fr3-arm-controller` alias dance. Those were split
# only because the controller spawner has to run AFTER controller_manager exists — not because
# they are independent things. They are one unit: restarting bringup tears down
# controller_manager, which drops the controller, so the spawner has to run again anyway.
#
# What is deliberately NOT in here: move_group and the two PolyUMI bridges. Those live in
# fr3_inference.launch.py, so this file can be restarted on its own — which matters, because
# per docs/crb-fr3-inference.md ("When it doesn't come up") this is the component that crashes
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
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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

# Cap on the set_load call below, whose own wait for the service is unbounded. Matched to the
# spawner's wait, since both are waiting on the same thing coming up: franka_bringup.
SET_LOAD_TIMEOUT_S = CONTROLLER_MANAGER_TIMEOUT_S


def generate_launch_description():
    """Include franka_bringup and spawn fr3_arm_controller once controller_manager is up."""
    robot_ip = LaunchConfiguration('robot_ip')
    arm_id = LaunchConfiguration('arm_id')
    load_gripper = LaunchConfiguration('load_gripper')

    # Tell the FCI what is bolted to the flange, so its gravity compensation cancels the whole
    # weight. Without this the unmodelled part is a constant force the cartesian impedance spring
    # fights, visible as the TCP dropping the moment that controller activates. See nuc/tcp_calib.py.
    #
    # This is the ONLY window in which it can be set: the FR3 refuses the command once a controller
    # holds the arm, with `Set Load command rejected: command not possible in the current mode
    # ("Move")`. Hence the ordering below, ahead of the spawner. Changing it later means deactivating
    # every controller first — see docs/calibration-instructions.md.
    #
    # Run through a shell so the exit status reflects the RESPONSE, not just the transport: on its
    # own `ros2 service call` exits 0 even when the body says `success: false`, and
    # franka_param_service_server flattens every CommandException to the string "command exception
    # error", so a rejected payload otherwise scrolls past as ordinary output. `timeout` because the
    # wait for the service is unbounded, and `tee` so the real response still reaches the log.
    set_load_cmd = (
        'timeout ' + SET_LOAD_TIMEOUT_S + ' ros2 service call /service_server/set_load '
        "franka_msgs/srv/SetLoad '" + tcp_calib.set_load_request() + "' "
        '| tee /dev/stderr | grep -q success=True'
    )
    set_load = ExecuteProcess(cmd=['bash', '-c', set_load_cmd], output='screen')

    # The joint-trajectory controller move_group executes through. franka.launch.py spawns
    # only the two broadcasters, so without this /execute_trajectory has nothing to drive
    # and the arm never moves. The two arguments do DIFFERENT jobs and both are required:
    # Humble's spawner sets the `type` param only from -t (spawner.py:208), while
    # --param-file goes through set_controller_parameters_from_param_files, which sets the
    # controller's `params_file` and never reads a type out of it. Dropping -t fails with
    # "The 'type' param was not defined for ...", pointing at the controller rather than at
    # the missing flag.
    #
    # Hoisted out of the LaunchDescription list so set_load can be ordered ahead of it: setLoad is a
    # libfranka command, and it lands cleanly only while no controller holds the effort interfaces.
    arm_controller_spawner = Node(
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
    )

    def on_set_load_exit(event, context):
        """
        Spawn fr3_arm_controller once set_load is done, shouting first if the payload did not take.

        The spawner runs either way. Refusing to spawn would leave the whole session dead rather
        than merely mis-modelled, and bringup is the piece that has to stay restartable — an
        unmodelled payload is at least visible as TCP droop once the impedance controller
        activates. What is not acceptable is that being quiet: by the time you notice the droop the
        controller is up, and setLoad can then only be retried by deactivating everything again.
        """
        if event.returncode == 0:
            return [arm_controller_spawner]
        return [
            LogInfo(
                msg=(
                    f'\n!!! SetLoad FAILED (exit {event.returncode}) — the arm is running with an '
                    'UNMODELLED PAYLOAD.\n'
                    '!!! Expect the TCP to sag when the impedance controller activates.\n'
                    '!!! The real reason is in the /service_server log, not the response above.\n'
                    '!!! Retrying means deactivating controllers first — '
                    'see docs/calibration-instructions.md.\n'
                )
            ),
            arm_controller_spawner,
        ]

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
            LogInfo(msg=f'[fr3_bringup] {tcp_calib.describe_payload()}'),
            set_load,
            # Ordered on set_load's exit, not declared alongside it: the spawner activates
            # fr3_arm_controller, which claims the effort interfaces and puts the robot in "Move"
            # mode — where setLoad is rejected outright. This ordering is load-bearing, not tidiness.
            RegisterEventHandler(OnProcessExit(target_action=set_load, on_exit=on_set_load_exit)),
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
            # With load_gripper false the URDF has no hand at all, so robot_state_publisher stops
            # emitting the whole subtree: polyumi_tcp above is orphaned (no observation on the
            # laptop) and fr3_hand_tcp disappears (the impedance controller refuses to activate).
            # Both failures point nowhere near this flag. Republish the fixed joints the URDF lost
            # — see nuc/tcp_calib.py for which, and why the finger joints are not among them.
            *[
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name=name,
                    output='screen',
                    arguments=args,
                    condition=UnlessCondition(load_gripper),
                )
                for name, args in tcp_calib.hand_transform_publishers()
            ],
        ]
    )

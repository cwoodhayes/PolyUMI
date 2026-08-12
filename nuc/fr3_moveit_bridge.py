#!/usr/bin/env python3
"""
FR3 MoveIt bridge — runs ON THE NUC (ROS 2 Humble).

Subscribes to target EEF pose chunks on /polyumi/target_poses (a PoseArray published by
the laptop's policy_client_node over DDS — one action chunk from the inference server) and
drives the arm via the LOCAL move_group, planning the whole chunk as a single multi-waypoint
Cartesian path. This node must run on the NUC, not the laptop: the laptop (rmw_cyclonedds
4.0.2, Kilted) and the NUC (rmw_cyclonedds 1.3.4, Humble) can exchange small messages like
PoseArray fine, but the large nested MoveIt action goals (MoveGroup.Goal /
GetCartesianPath.Request) get corrupted across the rmw-major boundary ("invalid data size,
at serdata.cpp:384" -> move_group "Catastrophic failure"). Keeping the move_group calls
same-rmw (NUC-local) avoids that.

Self-contained (no PolyUMI package deps) so it runs from a plain clone on the NUC:
    source /opt/ros/humble/setup.bash
    source ~/franka_ws/install/setup.bash   # move_group + franka must be up (fr3-bringup)
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file://$HOME/franka_ws/config/cyclonedds.xml
    python3 nuc/fr3_moveit_bridge.py --ros-args -p execute:=false   # dry-run (no motion)

Set execute:=true to actually move the arm. Default is false (plan only) for safety.

Also serves /polyumi/home (std_srvs/Trigger), a joint-space move to the SRDF `ready` pose.
Callable from the laptop despite the rmw gap, as long as the type is given explicitly (the ROS
*graph* does not cross Humble<->Kilted, so `ros2 node list` and node-name lookups come back
empty, but service calls match on DDS endpoints and work fine):

    ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"
"""

import math
import threading

from geometry_msgs.msg import Pose, PoseArray
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

# FR3 SRDF names. IMPORTANT: use group 'fr3_arm', NOT 'fr3_manipulator'. Only fr3_arm has
# an IK solver entry in kinematics.yaml, and Humble's computeCartesianPath needs it — with
# fr3_manipulator every Cartesian request returns fraction=0.0 (verified on hardware).
# fr3_arm still accepts an arbitrary target link (fraction=1.0), even though its SRDF tip is
# fr3_link8, so we keep controlling the true TCP.
#
# That target is polyumi_tcp, NOT the stock fr3_hand_tcp: incoming poses are the policy's, and
# the policy's body frame is the closed-fingertip midpoint in optical axes. See nuc/tcp_calib.py
# for the transform and where it comes from. move_group only knows the link because
# fr3_move_group.launch.py feeds it nuc/description/fr3_polyumi.urdf.xacro — planning against a
# stock franka_description will fail here with "Link 'polyumi_tcp' not found".
DEFAULT_GROUP = 'fr3_arm'
DEFAULT_LINK = 'polyumi_tcp'
DEFAULT_BASE = 'fr3_link0'

MIN_CARTESIAN_FRACTION = 0.9
PLAN_TIMEOUT_S = 5.0
# Chunks can be several waypoints at a low velocity scaling, so give execution more room
# than a single-waypoint move would need.
EXECUTE_TIMEOUT_S = 30.0
# How long to let a cancelled goal actually abort before giving up on it. Short: move_group stops
# the controller immediately, so this is deceleration, not motion.
CANCEL_TIMEOUT_S = 3.0

# --- Homing (the /polyumi/home service) ---
# The SRDF's own `ready` group state for fr3_arm (franka_fr3_moveit_config, group_definition.xacro).
# Overridable via the `home_joints` param when a task wants to start somewhere else.
HOME_JOINT_NAMES = [f'fr3_joint{i}' for i in range(1, 8)]
HOME_JOINTS = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
HOME_TOLERANCE_RAD = 0.01
HOME_PLAN_TIME_S = 5.0
# Homing is a joint-space sweep across the workspace, not a few-centimetre chunk, and it runs at
# the same max_velocity_scaling — a 3 s planned move at 0.1 takes 30 s. EXECUTE_TIMEOUT_S would
# abort it partway.
HOME_EXECUTE_TIMEOUT_S = 120.0


class Fr3MoveItBridge(Node):
    """Receive target EEF pose chunks and drive the FR3 via the local move_group."""

    def __init__(self, **kwargs):
        """Declare params, create the target-pose subscription and move_group clients."""
        super().__init__('fr3_moveit_bridge', **kwargs)

        self.declare_parameter('execute', False)
        self.declare_parameter('planning_group', DEFAULT_GROUP)
        self.declare_parameter('eef_link', DEFAULT_LINK)
        self.declare_parameter('base_frame', DEFAULT_BASE)
        self.declare_parameter('max_velocity_scaling', 0.1)
        self.declare_parameter('target_topic', '/polyumi/target_poses')
        self.declare_parameter('home_joints', HOME_JOINTS)

        self._execute = self.get_parameter('execute').get_parameter_value().bool_value
        self._group = self.get_parameter('planning_group').get_parameter_value().string_value
        self._link = self.get_parameter('eef_link').get_parameter_value().string_value
        self._base = self.get_parameter('base_frame').get_parameter_value().string_value
        self._vscale = self.get_parameter('max_velocity_scaling').get_parameter_value().double_value
        topic = self.get_parameter('target_topic').get_parameter_value().string_value

        self._home_joints = list(self.get_parameter('home_joints').get_parameter_value().double_array_value)

        self._cbgroup = ReentrantCallbackGroup()
        self._cartesian = self.create_client(GetCartesianPath, 'compute_cartesian_path', callback_group=self._cbgroup)
        self._joint_plan = self.create_client(GetMotionPlan, 'plan_kinematic_path', callback_group=self._cbgroup)
        self._exec = ActionClient(self, ExecuteTrajectory, 'execute_trajectory', callback_group=self._cbgroup)
        self.create_service(Trigger, '/polyumi/home', self._on_home, callback_group=self._cbgroup)

        # Latest-goal, skip-while-busy: drop poses that arrive while a plan/execute is in
        # flight so we always act on the freshest target without queuing up stale ones.
        self._busy = threading.Lock()

        self.create_subscription(PoseArray, topic, self._on_target, 10, callback_group=self._cbgroup)

        # Fail loudly at startup rather than on the first target pose.
        if not self._cartesian.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                'compute_cartesian_path NOT found after 10s — move_group is probably not running '
                'on this NUC. Start it first: ros2 launch nuc/launch/fr3_move_group.launch.py '
                'robot_ip:=192.168.51.20'
            )
        else:
            self.get_logger().info('move_group found (compute_cartesian_path ready).')

        if not self._exec.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'execute_trajectory action server NOT found after 10s — move_group is probably not '
                'running on this NUC. Start it first: ros2 launch nuc/launch/fr3_move_group.launch.py '
                'robot_ip:=192.168.51.20'
            )
        else:
            self.get_logger().info('move_group found (execute_trajectory ready).')

        mode = 'EXECUTE (arm will move)' if self._execute else 'plan-only (no motion)'
        self.get_logger().info(f'fr3_moveit_bridge started — listening on {topic} — mode: {mode}')
        self.get_logger().info(
            '/polyumi/home is up (std_srvs/Trigger). It MOVES THE ARM even in plan-only mode — '
            'it is an explicit request, unlike the streamed chunks `execute` guards.'
        )

    def _on_target(self, msg: PoseArray) -> None:
        """Plan (and optionally execute) a multi-waypoint Cartesian path through the chunk."""
        if not msg.poses:
            self.get_logger().warn('Received empty target pose chunk, ignoring.')
            return
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn(f'Dropped target chunk ({len(msg.poses)} poses): previous plan/execute in flight')
            return
        try:
            frame_id = msg.header.frame_id or self._base
            trajectory = self._plan_cartesian(list(msg.poses), frame_id)
            if trajectory is None:
                return
            if not self._execute:
                self.get_logger().info(f'Plan OK ({len(msg.poses)} waypoints, plan-only mode, not executing).')
                return
            if self._run_execute(trajectory):
                self.get_logger().info(f'Executed chunk ({len(msg.poses)} waypoints).')
        finally:
            self._busy.release()

    def _on_home(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """
        Move the arm to the home joint pose. THIS MOVES THE ARM, regardless of `execute`.

        That is deliberate. `execute` gates the *policy's* streamed chunks — the danger there is
        motion you did not ask for, arriving on a topic at 10 Hz. Calling a service named `home`
        is the opposite: an explicit, one-shot operator request. Gating it on `execute` would
        also make it a no-op in the default configuration, since fr3_inference.launch.py defaults
        execute_arm:=false, which is exactly when you most want to reposition the arm by hand.

        Joint-space, not Cartesian: a Cartesian path from an arbitrary pose back to home is
        happy to fail the fraction check, or to drag the gripper straight through the table.
        """
        if len(self._home_joints) != len(HOME_JOINT_NAMES):
            response.success = False
            response.message = f'home_joints has {len(self._home_joints)} values, expected {len(HOME_JOINT_NAMES)}'
            self.get_logger().error(response.message)
            return response
        if not self._busy.acquire(blocking=False):
            response.success = False
            response.message = 'busy: a plan/execute is already in flight'
            self.get_logger().warn(f'/polyumi/home refused — {response.message}')
            return response
        try:
            self.get_logger().warn('/polyumi/home called — MOVING THE ARM to the home pose.')
            trajectory = self._plan_to_joints(self._home_joints)
            if trajectory is None:
                response.success = False
                response.message = 'planning to the home pose failed — see the bridge log'
                return response
            if self._run_execute(trajectory, HOME_EXECUTE_TIMEOUT_S):
                response.success = True
                response.message = 'homed'
                self.get_logger().info('Homed.')
            else:
                response.success = False
                response.message = 'execution failed — see the bridge log'
            return response
        finally:
            self._busy.release()

    def _plan_to_joints(self, positions: list[float]) -> RobotTrajectory | None:
        """Plan a collision-checked joint-space move to `positions`; return the trajectory or None."""
        if not self._joint_plan.service_is_ready():
            self.get_logger().error(
                'plan_kinematic_path is NOT available — is move_group running on this NUC? '
                '(ros2 launch nuc/launch/fr3_move_group.launch.py robot_ip:=192.168.51.20)'
            )
            return None

        req = GetMotionPlan.Request()
        mpr = req.motion_plan_request
        mpr.group_name = self._group
        mpr.num_planning_attempts = 10
        mpr.allowed_planning_time = HOME_PLAN_TIME_S
        # Scaling is left at the planner's default and applied by _slow_trajectory instead, so
        # max_velocity_scaling stays the single speed knob for both this and the Cartesian path.
        goal = Constraints()
        for name, position in zip(HOME_JOINT_NAMES, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = position
            jc.tolerance_above = HOME_TOLERANCE_RAD
            jc.tolerance_below = HOME_TOLERANCE_RAD
            jc.weight = 1.0
            goal.joint_constraints.append(jc)
        mpr.goal_constraints.append(goal)

        future = self._joint_plan.call_async(req)
        if not self._wait(future, HOME_PLAN_TIME_S + PLAN_TIMEOUT_S):
            self.get_logger().warning('Joint-space planning timed out.')
            return None
        resp = future.result()
        if resp is None or resp.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if resp is None else resp.motion_plan_response.error_code.val
            self.get_logger().warning(f'Joint-space planning failed (error_code={code}).')
            return None
        return resp.motion_plan_response.trajectory

    def _plan_cartesian(self, poses: list[Pose], frame_id: str) -> RobotTrajectory | None:
        """Request a multi-waypoint Cartesian path through poses; return the trajectory or None."""
        req = GetCartesianPath.Request()
        req.header.frame_id = frame_id
        req.group_name = self._group
        req.link_name = self._link
        req.max_step = 0.01
        # NOTE: Humble's GetCartesianPath.Request has NO max_velocity/acceleration_scaling_factor
        # fields (added in a later MoveIt). Speed is limited via the trajectory time
        # parameterization instead — see _slow_trajectory() applied before execution.
        req.waypoints = poses

        # Check the service is actually there first: call_async on a missing service never
        # completes, which would surface as a mystifying "planning timed out".
        if not self._cartesian.service_is_ready():
            self.get_logger().error(
                'compute_cartesian_path is NOT available — is move_group running on this NUC? '
                '(ros2 launch nuc/launch/fr3_move_group.launch.py robot_ip:=192.168.51.20)'
            )
            return None

        future = self._cartesian.call_async(req)
        if not self._wait(future, PLAN_TIMEOUT_S):
            self.get_logger().warning('Cartesian planning timed out.')
            return None
        resp = future.result()
        if resp is None or resp.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if resp is None else resp.error_code.val
            self.get_logger().warning(f'Cartesian planning failed (error_code={code}).')
            return None
        if resp.fraction < MIN_CARTESIAN_FRACTION:
            self.get_logger().warning(f'Cartesian plan only {resp.fraction:.0%} complete, skipping.')
            return None
        return resp.solution

    def _slow_trajectory(self, trajectory: RobotTrajectory) -> RobotTrajectory:
        """
        Scale a trajectory in time to cap end-effector speed.

        Humble's GetCartesianPath has no velocity_scaling field, so we stretch
        time_from_start by 1/vscale and scale velocities/accelerations down — the same
        path, run proportionally slower.

        move_group already time-parameterizes the Cartesian path at full speed against the
        URDF joint limits, so scale=1.0 means "as fast as MoveIt planned" and is the
        ceiling: scaling above 1.0 would compress time below the planned profile and
        exceed those limits, so it is clamped.
        """
        scale = min(max(self._vscale, 1e-3), 1.0)
        if self._vscale > 1.0:
            self.get_logger().warn(
                f'max_velocity_scaling={self._vscale} > 1.0 would exceed the planned joint '
                'limits; clamping to 1.0 (already full planned speed).'
            )
        jt = trajectory.joint_trajectory
        # Track the last point's time while iterating rather than indexing jt.points[-1]
        # afterward: the rosidl array-field type stub doesn't support __getitem__.
        last_time = None
        for pt in jt.points:
            total_ns = pt.time_from_start.sec * 1_000_000_000 + pt.time_from_start.nanosec
            total_ns = int(total_ns / scale)
            pt.time_from_start.sec = total_ns // 1_000_000_000
            pt.time_from_start.nanosec = total_ns % 1_000_000_000
            pt.velocities = [v * scale for v in pt.velocities]
            pt.accelerations = [a * scale * scale for a in pt.accelerations]
            last_time = pt.time_from_start
        if last_time is not None:
            self.get_logger().info(
                f'Executing {len(jt.points)} pts over {last_time.sec + last_time.nanosec / 1e9:.2f}s '
                f'(vscale={scale:g}; raise it to go faster, 1.0 = full planned speed)'
            )
        return trajectory

    def _run_execute(self, trajectory: RobotTrajectory, timeout_s: float = EXECUTE_TIMEOUT_S) -> bool:
        """Execute a planned trajectory via ExecuteTrajectory; block until done."""
        # Same rationale as _plan_cartesian's service_is_ready() check: send_goal_async on a
        # server that isn't there hangs until PLAN_TIMEOUT_S instead of failing immediately,
        # holding the busy lock and dropping chunks that arrive in the meantime.
        if not self._exec.server_is_ready():
            self.get_logger().error(
                'execute_trajectory action server is NOT available — is move_group running on this '
                'NUC? (ros2 launch nuc/launch/fr3_move_group.launch.py robot_ip:=192.168.51.20)'
            )
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = self._slow_trajectory(trajectory)
        gf = self._exec.send_goal_async(goal)
        if not self._wait(gf, PLAN_TIMEOUT_S):
            self.get_logger().warning('Execute goal submission timed out.')
            return False
        gh = gf.result()
        if gh is None or not gh.accepted:
            self.get_logger().warning('Execute goal rejected.')
            return False
        rf = gh.get_result_async()
        if not self._wait(rf, timeout_s):
            # Cancel, and wait for the abort to land. Returning here releases the caller's busy
            # lock, and an uncancelled goal keeps driving the arm — so the next chunk would plan
            # a Cartesian path from a start state the arm has already left. Bounded, because a
            # server that ignores the cancel must not wedge the bridge instead.
            self.get_logger().warning(f'Execution timed out after {timeout_s:.0f}s — cancelling.')
            gh.cancel_goal_async()
            if not self._wait(rf, CANCEL_TIMEOUT_S):
                self.get_logger().error(
                    'Cancel did not take effect — the arm may still be moving. Stop it from the '
                    'Desk UI before sending anything else.'
                )
            return False
        res = rf.result()
        if res is None or res.result.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if res is None else res.result.error_code.val
            self.get_logger().warning(f'Execution failed (error_code={code}).')
            return False
        return True

    @staticmethod
    def _wait(future, timeout_s: float) -> bool:
        """Block until future completes or times out (node spins under a MultiThreadedExecutor)."""
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        return done.wait(timeout=timeout_s)


def main():
    """Spin the bridge node under a multithreaded executor."""
    rclpy.init()
    node = Fr3MoveItBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

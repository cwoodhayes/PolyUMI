#!/usr/bin/env python3
"""
FR3 MoveIt bridge — runs ON THE NUC (ROS 2 Humble).

Serves /polyumi/home (std_srvs/Trigger): a joint-space move to the SRDF `ready` pose, planned
and executed through the LOCAL move_group. This node must run on the NUC, not the laptop: the
laptop (rmw_cyclonedds 4.0.2, Kilted) and the NUC (rmw_cyclonedds 1.3.4, Humble) can exchange
small messages fine, but the large nested MoveIt action goals (MoveGroup.Goal /
ExecuteTrajectory.Goal) get corrupted across the rmw-major boundary ("invalid data size, at
serdata.cpp:384" -> move_group "Catastrophic failure"). Keeping the move_group calls same-rmw
(NUC-local) avoids that.

Self-contained (no PolyUMI package deps) so it runs from a plain clone on the NUC:
    source /opt/ros/humble/setup.bash
    source ~/franka_ws/install/setup.bash   # move_group + franka must be up (fr3-bringup)
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file://$HOME/franka_ws/config/cyclonedds.xml
    python3 nuc/fr3_moveit_bridge.py --ros-args -p max_velocity_scaling:=0.1

Callable from the laptop despite the rmw gap, as long as the type is given explicitly (the ROS
*graph* does not cross Humble<->Kilted, so `ros2 node list` and node-name lookups come back
empty, but service calls match on DDS endpoints and work fine):

    ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"
"""

import math
import threading

from controller_manager_msgs.srv import SwitchController
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetMotionPlan
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

# The SRDF group homing plans in.
DEFAULT_GROUP = 'fr3_arm'

PLAN_TIMEOUT_S = 5.0
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
# the same max_velocity_scaling — a 3 s planned move at 0.1 takes 30 s.
HOME_EXECUTE_TIMEOUT_S = 120.0

# --- Controller handover ---
# move_group executes through fr3_arm_controller; the streaming impedance controller claims the
# same <joint>/effort interfaces, so the two are mutually exclusive and homing has to borrow the
# arm back. Switching restarts the libfranka control loop (franka_hardware's
# perform_command_mode_switch calls stopRobot() then re-initialises), so it must only happen with
# the arm stationary — which, at the start and end of a home, it is.
SERVO_CONTROLLER = 'polyumi_cartesian_impedance_controller'
MOVEIT_CONTROLLER = 'fr3_arm_controller'
SWITCH_TIMEOUT_S = 5.0


class Fr3MoveItBridge(Node):
    """Serve /polyumi/home, driving the FR3 to its SRDF ready pose via the local move_group."""

    def __init__(self, **kwargs):
        """Declare params, create the home service and move_group clients."""
        super().__init__('fr3_moveit_bridge', **kwargs)

        self.declare_parameter('planning_group', DEFAULT_GROUP)
        self.declare_parameter('max_velocity_scaling', 0.1)
        self.declare_parameter('home_joints', HOME_JOINTS)

        self._group = self.get_parameter('planning_group').get_parameter_value().string_value
        self._vscale = self.get_parameter('max_velocity_scaling').get_parameter_value().double_value

        self._home_joints = list(self.get_parameter('home_joints').get_parameter_value().double_array_value)

        self._cbgroup = ReentrantCallbackGroup()
        self._joint_plan = self.create_client(GetMotionPlan, 'plan_kinematic_path', callback_group=self._cbgroup)
        self._exec = ActionClient(self, ExecuteTrajectory, 'execute_trajectory', callback_group=self._cbgroup)
        self._switch = self.create_client(
            SwitchController, '/controller_manager/switch_controller', callback_group=self._cbgroup
        )
        self.create_service(Trigger, '/polyumi/home', self._on_home, callback_group=self._cbgroup)

        # Latest-goal, skip-while-busy: a concurrent /polyumi/home call while one is already
        # planning/executing is dropped rather than queued.
        self._busy = threading.Lock()

        # Fail loudly at startup rather than on the first /polyumi/home call.
        if not self._joint_plan.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                'plan_kinematic_path NOT found after 10s — move_group is probably not running '
                'on this NUC. Start it first: ros2 launch nuc/launch/fr3_move_group.launch.py '
                'robot_ip:=192.168.51.20'
            )
        else:
            self.get_logger().info('move_group found (plan_kinematic_path ready).')

        if not self._exec.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'execute_trajectory action server NOT found after 10s — move_group is probably not '
                'running on this NUC. Start it first: ros2 launch nuc/launch/fr3_move_group.launch.py '
                'robot_ip:=192.168.51.20'
            )
        else:
            self.get_logger().info('move_group found (execute_trajectory ready).')

        self.get_logger().info('fr3_moveit_bridge started — /polyumi/home is up (std_srvs/Trigger).')

    def _on_home(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """
        Move the arm to the home joint pose. THIS MOVES THE ARM.

        An explicit, one-shot operator request, unconditional on anything else — there is no
        streamed-chunk path left for it to be confused with.

        Joint-space, not Cartesian: a Cartesian path from an arbitrary pose back to home is
        happy to drag the gripper straight through the table.
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
            # Borrow the arm from the streaming controller if it holds it. Not conditional on
            # having seen it start: this bridge and the controller come up independently, and a
            # switch naming an inactive controller is a no-op, so asking is cheaper than tracking.
            handed_over = self._switch_controllers(activate=MOVEIT_CONTROLLER, deactivate=SERVO_CONTROLLER)
            try:
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
                # Hand the arm back only if we took it. Leaving the servo deactivated after a
                # failed home would look like the policy silently doing nothing — and that
                # includes a home that itself SUCCEEDED: this must still turn a true `homed`
                # response into a false one if the hand-back is what failed. Mutating `response`
                # here reaches the caller even though `return response` above already fired —
                # Python evaluates that expression to the object reference first, then runs this
                # block, then returns the (now possibly-mutated) object.
                if handed_over and not self._switch_controllers(
                    activate=SERVO_CONTROLLER, deactivate=MOVEIT_CONTROLLER
                ):
                    response.success = False
                    response.message += (
                        f' — but failed to hand the arm back to {SERVO_CONTROLLER}; it is still on '
                        f'{MOVEIT_CONTROLLER} and the policy cannot drive it. Retry the switch '
                        'manually.'
                    )
        finally:
            self._busy.release()

    def _switch_controllers(self, *, activate: str, deactivate: str) -> bool:
        """
        Swap which controller drives the arm; return whether the switch actually happened.

        STRICT, not BEST_EFFORT: a partial switch would leave the arm with either two controllers
        claiming its effort interfaces or none at all, and the second is indistinguishable from a
        working system that has simply stopped moving.

        A False return is not always an error — asking to deactivate a controller that was never
        spawned fails the same way, which is the normal case when running without the streaming
        controller at all.
        """
        if not self._switch.wait_for_service(timeout_sec=SWITCH_TIMEOUT_S):
            self.get_logger().warn(
                'controller_manager switch_controller not available; leaving controllers as they are.'
            )
            return False

        request = SwitchController.Request()
        request.activate_controllers = [activate]
        request.deactivate_controllers = [deactivate]
        request.strictness = SwitchController.Request.STRICT
        future = self._switch.call_async(request)
        if not self._wait(future, SWITCH_TIMEOUT_S):
            self.get_logger().error(f'switch to {activate} timed out after {SWITCH_TIMEOUT_S}s')
            return False

        ok = bool(future.result() and future.result().ok)
        if ok:
            self.get_logger().info(f'Controller switched: {deactivate} -> {activate}')
        else:
            self.get_logger().info(
                f'Controller switch {deactivate} -> {activate} declined; '
                f'{deactivate} is probably not running. Continuing.'
            )
        return ok

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
        # max_velocity_scaling stays the single speed knob.
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

    def _slow_trajectory(self, trajectory: RobotTrajectory) -> RobotTrajectory:
        """
        Scale a trajectory in time to cap end-effector speed.

        plan_kinematic_path has no velocity_scaling field on Humble, so we stretch
        time_from_start by 1/vscale and scale velocities/accelerations down — the same
        path, run proportionally slower.

        move_group already time-parameterizes the plan at full speed against the URDF joint
        limits, so scale=1.0 means "as fast as MoveIt planned" and is the ceiling: scaling
        above 1.0 would compress time below the planned profile and exceed those limits, so
        it is clamped.
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

    def _run_execute(self, trajectory: RobotTrajectory, timeout_s: float) -> bool:
        """Execute a planned trajectory via ExecuteTrajectory; block until done."""
        # Same rationale as _plan_to_joints's service_is_ready() check: send_goal_async on a
        # server that isn't there hangs until PLAN_TIMEOUT_S instead of failing immediately,
        # holding the busy lock past when the caller would otherwise give up.
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
            # lock, and an uncancelled goal keeps driving the arm — so a follow-up /polyumi/home
            # would plan from a start state the arm has already left. Bounded, because a server
            # that ignores the cancel must not wedge the bridge instead.
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

"""
MoveIt2 Cartesian-execution client for the FR3, over DDS.

Talks to a `move_group` running on the FR3 NUC (Humble) purely through raw
`moveit_msgs` action/service clients — no `moveit_py`, no local robot_description.
This mirrors the proven pattern from the penpal project (which drove a Franka via
MoveIt-from-Python on this same PC) and keeps all PolyUMI code on the Kilted laptop.

The single entry point `plan_and_execute_cartesian(pose_xyzquat)` is **synchronous
and blocking** so it drops straight into the policy client's control tick: it plans
a one-waypoint Cartesian path to the target EEF pose via `/compute_cartesian_path`,
then executes the returned trajectory via the `/execute_trajectory` action. All
clients sit on a dedicated reentrant callback group so their futures resolve on
other executor threads while the calling tick blocks (the node must run under a
`MultiThreadedExecutor`).
"""

import threading

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

# FR3 SRDF names (see docs/crb-fr3-inference.md). fr3_manipulator's tip is
# fr3_hand_tcp, matching the policy client's eef_frame. Cartesian planning uses a
# Jacobian, so the missing IK-solver entry for fr3_manipulator is not a problem here.
DEFAULT_PLANNING_GROUP = 'fr3_manipulator'
DEFAULT_EEF_LINK = 'fr3_hand_tcp'
DEFAULT_BASE_FRAME = 'fr3_link0'

# Minimum accepted Cartesian-path completion fraction (0..1). Below this the plan is
# too incomplete to execute safely, so we skip the target and warn.
MIN_CARTESIAN_FRACTION = 0.9

# How long to wait for a single plan or execute to finish before giving up on this
# target (seconds). Keeps a stuck move_group from blocking the control loop forever.
PLAN_TIMEOUT_S = 1.0
EXECUTE_TIMEOUT_S = 4.0


class FR3MoveItClient:
    """Plan and execute Cartesian EEF targets on the FR3 via a remote move_group."""

    def __init__(
        self,
        node: Node,
        planning_group: str = DEFAULT_PLANNING_GROUP,
        eef_link: str = DEFAULT_EEF_LINK,
        base_frame: str = DEFAULT_BASE_FRAME,
        max_velocity_scaling: float = 0.1,
        max_acceleration_scaling: float = 0.1,
    ):
        """Create the Cartesian-path service client and trajectory-execution action client."""
        self._node = node
        self._logger = node.get_logger()
        self._planning_group = planning_group
        self._eef_link = eef_link
        self._base_frame = base_frame
        self._max_velocity_scaling = max_velocity_scaling
        self._max_acceleration_scaling = max_acceleration_scaling

        # Reentrant group so goal-response / result / service futures can resolve on
        # other executor threads while the (blocking) control tick waits on them.
        self._cbgroup = ReentrantCallbackGroup()
        self._cartesian_client = node.create_client(
            GetCartesianPath, 'compute_cartesian_path', callback_group=self._cbgroup
        )
        self._execute_client = ActionClient(node, ExecuteTrajectory, 'execute_trajectory', callback_group=self._cbgroup)

    def wait_for_server(self, timeout_s: float = 10.0) -> bool:
        """Block until both move_group interfaces are discovered, or timeout."""
        if not self._cartesian_client.wait_for_service(timeout_sec=timeout_s):
            self._logger.error('compute_cartesian_path service not available — is move_group running on the NUC?')
            return False
        if not self._execute_client.wait_for_server(timeout_sec=timeout_s):
            self._logger.error('execute_trajectory action server not available (move_group).')
            return False
        return True

    def plan_and_execute_cartesian(self, pose_xyzquat, execute: bool = True) -> bool:
        """
        Plan a one-waypoint Cartesian path to the target EEF pose and (optionally) execute it.

        Args:
            pose_xyzquat: iterable [x, y, z, qx, qy, qz, qw] target in the base frame.
            execute: if False, plan only (dry run) — returns True on a successful plan
                without moving the robot.

        Returns:
            True if the target was planned (and executed, when execute=True) successfully;
            False if planning/execution failed or was too incomplete to run.

        """
        solution = self._plan_cartesian(pose_xyzquat)
        if solution is None:
            return False
        if not execute:
            return True
        return self._execute(solution)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _plan_cartesian(self, pose_xyzquat):
        """Request a Cartesian path to the single target pose; return the trajectory or None."""
        request = GetCartesianPath.Request()
        request.header.frame_id = self._base_frame
        request.group_name = self._planning_group
        request.link_name = self._eef_link
        request.max_step = 0.01
        request.max_velocity_scaling_factor = self._max_velocity_scaling
        request.max_acceleration_scaling_factor = self._max_acceleration_scaling

        pose = Pose()
        pose.position.x = float(pose_xyzquat[0])
        pose.position.y = float(pose_xyzquat[1])
        pose.position.z = float(pose_xyzquat[2])
        pose.orientation.x = float(pose_xyzquat[3])
        pose.orientation.y = float(pose_xyzquat[4])
        pose.orientation.z = float(pose_xyzquat[5])
        pose.orientation.w = float(pose_xyzquat[6])
        request.waypoints = [pose]

        future = self._cartesian_client.call_async(request)
        if not self._wait_for_future(future, PLAN_TIMEOUT_S):
            self._logger.warning('Cartesian path planning timed out.')
            return None

        response = future.result()
        if response is None:
            self._logger.warning('Cartesian path planning returned no response.')
            return None
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self._logger.warning(f'Cartesian planning failed (error_code={response.error_code.val}).')
            return None
        if response.fraction < MIN_CARTESIAN_FRACTION:
            self._logger.warning(
                f'Cartesian plan only {response.fraction:.0%} complete '
                f'(< {MIN_CARTESIAN_FRACTION:.0%}), skipping target.'
            )
            return None
        return response.solution

    def _execute(self, trajectory) -> bool:
        """Execute a planned trajectory via the ExecuteTrajectory action; block until done."""
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        goal_future = self._execute_client.send_goal_async(goal)
        if not self._wait_for_future(goal_future, PLAN_TIMEOUT_S):
            self._logger.warning('ExecuteTrajectory goal submission timed out.')
            return False

        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._logger.warning('ExecuteTrajectory goal was rejected by move_group.')
            return False

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(result_future, EXECUTE_TIMEOUT_S):
            self._logger.warning('ExecuteTrajectory execution timed out.')
            return False

        result = result_future.result()
        if result is None or result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            code = None if result is None else result.result.error_code.val
            self._logger.warning(f'Trajectory execution failed (error_code={code}).')
            return False
        return True

    def _wait_for_future(self, future, timeout_s: float) -> bool:
        """
        Block the calling thread until `future` completes or `timeout_s` elapses.

        Relies on the node running under a MultiThreadedExecutor so the future can be
        resolved on another thread while this one waits. Uses an on-done callback +
        Event rather than spin_until_future_complete, which would deadlock when called
        from inside an executor callback.
        """
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        return done.wait(timeout=timeout_s)

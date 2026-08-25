// Copyright (c) 2026 PolyUMI. MIT.

#pragma once

#include <polyumi_fr3_controllers/cartesian_impedance_law.hpp>
#include <polyumi_fr3_controllers/pose_trajectory_interpolator.hpp>

#include <controller_interface/controller_interface.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace polyumi_fr3_controllers {

/**
 * Streaming Cartesian impedance controller for the FR3, driven by absolutely-timed pose chunks.
 *
 * This is PolyUMI's on-arm executor. It replaces plan-then-execute (`fr3_moveit_bridge` calling
 * `compute_cartesian_path` + `ExecuteTrajectory`), which started every action chunk from rest,
 * stopped at its end, and discarded the policy's `dt` timeline — costing ~0.6 s of latency and all
 * compliance.
 *
 * Architecture, following UMI:
 *
 *   MultiDOFJointTrajectory (10 Hz, absolute times)   [non-realtime subscription]
 *     -> PoseTrajectoryInterpolator::scheduleWaypoint  splices without stopping
 *       -> update() at 1 kHz evaluates it              [realtime; the eval allocates nothing]
 *         -> Cartesian impedance law -> 7 joint torques
 *
 * The law itself is NOT allocation-free: nullspaceTorque runs a dynamic-size JacobiSVD every
 * cycle. Inherited from SERL, which ships the same thing in a 1 kHz franka_ros controller.
 *
 * The control law is SERL's (see cartesian_impedance_law.hpp), which is polymetis's — what UMI
 * actually deploys — plus an error clip and a nullspace term.
 *
 * The control point is `polyumi_tcp`, NOT franka's `O_T_EE` (which is `fr3_hand_tcp`, verified on
 * hardware). Both the measured pose and the Jacobian are moved onto it at activation using a TF
 * lookup, so `nuc/tcp_calib.py` stays the single definition of that frame.
 *
 * Torque, and not `franka_hardware`'s native `cartesian_pose` interface, which looks like the
 * direct analogue of UMI's `update_desired_ee_pose`: under it libfranka hardcodes
 * `ControllerMode::kJointImpedance` (`robot.cpp`), whose gains are stiffness-only with no damping
 * knob at all. Contact tasks need a real Cartesian mass-spring-damper. That interface also applies
 * no continuity safety net (the rate limiter and low-pass filter are hardcoded off), so nothing
 * downstream will catch a discontinuity for us either way — hence the interpolator is mandatory
 * and activation MUST seed from the measured pose.
 */
class CartesianImpedanceController : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  static constexpr int kNumJoints = 7;
  /// libfranka rejects torque jumps; this is SERL's per-cycle bound at 1 kHz.
  static constexpr double kMaxTorqueRate = 1.0;

  /// Everything tunable while the arm is live, snapshotted so the realtime loop reads it coherently.
  struct Gains {
    Matrix6d stiffness{Matrix6d::Zero()};
    Matrix6d damping{Matrix6d::Zero()};
    Matrix6d ki{Matrix6d::Zero()};
    ErrorClip clip;
    double nullspace_stiffness{0.0};
    double joint1_nullspace_stiffness{0.0};
    double max_pos_speed{0.0};
    double max_rot_speed{0.0};
  };

  void declareParameters();
  Gains readGainParameters();
  /// Push the configured collision thresholds to the robot. False if they did not take.
  bool applyCollisionBehavior();
  /// Measured TCP pose and the Jacobian at the TCP, both in the base frame.
  void readState(Pose& pose, Jacobian& jacobian, Vector7d& q, Vector7d& dq);
  void onTrajectory(const trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg);

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> robot_model_;
  rclcpp::Client<franka_msgs::srv::SetFullCollisionBehavior>::SharedPtr collision_client_;

  // Resolved by name at activation rather than indexed positionally into state_interfaces_. The
  // franka examples index by hardcoded offset, which silently reads the wrong joint the moment the
  // interface list changes — and reading the wrong joint here means commanding the wrong torque.
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> position_interfaces_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> velocity_interfaces_;
  std::vector<std::string> joint_names_;

  rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr target_sub_;
  // The interpolator is rebuilt (which allocates) in the subscription callback and handed to the
  // realtime loop by pointer. update() only ever reads and evaluates.
  realtime_tools::RealtimeBuffer<std::shared_ptr<const PoseTrajectoryInterpolator>> interpolator_;
  /// Gates onTrajectory. The subscription lives from on_configure, so without this a chunk
  /// arriving while the arm is on loan to move_group (a /polyumi/home) splices against a
  /// now_seconds_ that update() stopped advancing, growing the trajectory for the whole home.
  std::atomic<bool> active_{false};
  /// Written by update(), read by the subscription callback so its splice knows "now".
  std::atomic<double> now_seconds_{0.0};
  /// Latest scheduled waypoint time; the splice window's upper bound. Callback-thread only.
  double last_waypoint_time_{0.0};

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  /// Base-frame translation from franka's O_T_EE origin to polyumi_tcp. Resolved at activation.
  Eigen::Vector3d ee_to_tcp_{Eigen::Vector3d::Zero()};
  /// Rotation from the O_T_EE frame to polyumi_tcp. Fixed; also resolved at activation.
  Eigen::Quaterniond ee_to_tcp_rotation_{Eigen::Quaterniond::Identity()};

  std::string arm_id_;
  std::string base_frame_;
  std::string tcp_frame_;

  // Re-read off the realtime thread and handed over by pointer, so gains can be tuned against a
  // real contact without deactivating the controller — which would stop and restart the libfranka
  // loop between every attempt.
  realtime_tools::RealtimeBuffer<Gains> gains_;
  rclcpp::TimerBase::SharedPtr gain_refresh_timer_;

  Vector7d q_nullspace_{Vector7d::Zero()};
  Vector6d integral_error_{Vector6d::Zero()};
  /// Last torque we commanded, which is what saturateTorqueRate bounds against. Starts at zero,
  /// which is also what a zero-error activation commands, so there is nothing to seed.
  Vector7d tau_previous_{Vector7d::Zero()};
};

}  // namespace polyumi_fr3_controllers

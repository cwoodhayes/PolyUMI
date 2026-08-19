// Copyright (c) 2026 PolyUMI. MIT.

#include <polyumi_fr3_controllers/cartesian_impedance_controller.hpp>

#include <controller_interface/helpers.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace polyumi_fr3_controllers {

namespace {

constexpr char kRobotModelInterface[] = "robot_model";
constexpr char kRobotStateInterface[] = "robot_state";
constexpr char kCollisionService[] = "/service_server/set_full_collision_behavior";
constexpr std::chrono::seconds kCollisionTimeout{5};

Pose poseFromColumnMajor(const std::array<double, 16>& m) {
  const Eigen::Affine3d transform(Eigen::Matrix4d::Map(m.data()));
  Pose pose;
  pose.position = transform.translation();
  pose.orientation = Eigen::Quaterniond(transform.linear());
  return pose;
}

/// Smallest speed limit that still behaves like a limit rather than a division by zero.
constexpr double kMinSpeed = 1e-6;

Matrix6d diagonalGain(double translational, double rotational) {
  Matrix6d gain = Matrix6d::Identity();
  gain.topLeftCorner(3, 3) *= translational;
  gain.bottomRightCorner(3, 3) *= rotational;
  return gain;
}

double clampedSpeed(const std::shared_ptr<rclcpp_lifecycle::LifecycleNode>& node,
                    const char* name) {
  const double value = node->get_parameter(name).as_double();
  if (value >= kMinSpeed) {
    return value;
  }
  RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 5000,
                       "%s is %g; clamping to %g. A non-positive speed limit makes every spliced "
                       "waypoint land at infinity and the arm holds silently.",
                       name, value, kMinSpeed);
  return kMinSpeed;
}

}  // namespace

// ---------------------------------------------------------------------------
// Interface configuration
// ---------------------------------------------------------------------------

controller_interface::InterfaceConfiguration
CartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CartesianImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
  }
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  for (const auto& name : robot_model_->get_state_interface_names()) {
    config.names.push_back(name);
  }
  return config;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

CallbackReturn CartesianImpedanceController::on_init() {
  try {
    declareParameters();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to declare parameters: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

void CartesianImpedanceController::declareParameters() {
  // auto_declare, not declare_parameter: controller_manager loads the spawner's --param-file into
  // this node BEFORE on_init() runs, so every parameter named in the yaml is already declared and
  // a plain declare_parameter throws ParameterAlreadyDeclaredException. auto_declare returns the
  // already-loaded value in that case, which is what we want — the file should win over these
  // defaults. The defaults exist so the controller still runs with no param file at all.
  auto_declare<std::string>("arm_id", "fr3");
  auto_declare<std::string>("base_frame", "fr3_link0");
  auto_declare<std::string>("tcp_frame", "polyumi_tcp");
  auto_declare<std::string>("target_topic", "/polyumi/target_poses_traj");

  // SERL's shipped defaults. 2000 N/m with a 0.01 m clip is ~20 N of commanded force: stiff enough
  // to track, soft enough to lean on. See docs/franka-inference-bringup.md for why not UMI's.
  auto_declare<double>("translational_stiffness", 2000.0);
  auto_declare<double>("translational_damping", 89.0);
  auto_declare<double>("rotational_stiffness", 150.0);
  auto_declare<double>("rotational_damping", 7.0);
  auto_declare<double>("translational_ki", 0.0);
  auto_declare<double>("rotational_ki", 0.0);
  auto_declare<double>("translational_clip", 0.01);
  auto_declare<double>("rotational_clip", 0.05);
  auto_declare<double>("nullspace_stiffness", 0.2);
  auto_declare<double>("joint1_nullspace_stiffness", 100.0);

  // Bound how fast the equilibrium point may travel, which bounds how far it can lead the arm.
  auto_declare<double>("max_pos_speed", 1.0);
  auto_declare<double>("max_rot_speed", 3.14);

  // Collision reflex thresholds. Defaults are franka_example_controllers' own
  // DefaultRobotBehavior values, which every franka example applies in on_configure. The robot's
  // factory defaults are lower, and `cartesian_reflex` does not degrade gracefully — it aborts the
  // motion and takes ros2_control_node down with it.
  auto_declare<std::vector<double>>("collision.lower_torque_thresholds",
                                    {25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0});
  auto_declare<std::vector<double>>("collision.upper_torque_thresholds",
                                    {35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0});
  auto_declare<std::vector<double>>("collision.lower_force_thresholds",
                                    {30.0, 30.0, 30.0, 25.0, 25.0, 25.0});
  auto_declare<std::vector<double>>("collision.upper_force_thresholds",
                                    {40.0, 40.0, 40.0, 35.0, 35.0, 35.0});
}

bool CartesianImpedanceController::applyCollisionBehavior() {
  auto node = get_node();
  const auto lower_torque = node->get_parameter("collision.lower_torque_thresholds").as_double_array();
  const auto upper_torque = node->get_parameter("collision.upper_torque_thresholds").as_double_array();
  const auto lower_force = node->get_parameter("collision.lower_force_thresholds").as_double_array();
  const auto upper_force = node->get_parameter("collision.upper_force_thresholds").as_double_array();

  if (lower_torque.size() != 7 || upper_torque.size() != 7 || lower_force.size() != 6 ||
      upper_force.size() != 6) {
    RCLCPP_FATAL(node->get_logger(),
                 "collision.* thresholds must be 7 torques and 6 forces; got %zu/%zu torque and "
                 "%zu/%zu force values.",
                 lower_torque.size(), upper_torque.size(), lower_force.size(), upper_force.size());
    return false;
  }

  if (!collision_client_->wait_for_service(kCollisionTimeout)) {
    RCLCPP_FATAL(node->get_logger(), "%s not available after 5s — is fr3_bringup running?",
                 kCollisionService);
    return false;
  }

  auto request = std::make_shared<franka_msgs::srv::SetFullCollisionBehavior::Request>();
  // Nominal and acceleration get the same values, as DefaultRobotBehavior does. Splitting them
  // only matters if you want to be more permissive while accelerating, which we do not.
  std::copy_n(lower_torque.begin(), 7, request->lower_torque_thresholds_nominal.begin());
  std::copy_n(upper_torque.begin(), 7, request->upper_torque_thresholds_nominal.begin());
  std::copy_n(lower_torque.begin(), 7, request->lower_torque_thresholds_acceleration.begin());
  std::copy_n(upper_torque.begin(), 7, request->upper_torque_thresholds_acceleration.begin());
  std::copy_n(lower_force.begin(), 6, request->lower_force_thresholds_nominal.begin());
  std::copy_n(upper_force.begin(), 6, request->upper_force_thresholds_nominal.begin());
  std::copy_n(lower_force.begin(), 6, request->lower_force_thresholds_acceleration.begin());
  std::copy_n(upper_force.begin(), 6, request->upper_force_thresholds_acceleration.begin());

  // Blocking is safe here: ros2_control_node spins a MultiThreadedExecutor, so another thread
  // services the response while this one waits. The franka example controllers do the same.
  auto future = collision_client_->async_send_request(request);
  if (future.wait_for(kCollisionTimeout) != std::future_status::ready) {
    RCLCPP_FATAL(node->get_logger(), "%s did not answer within 5s.", kCollisionService);
    return false;
  }

  const auto response = future.get();
  if (!response->success) {
    RCLCPP_FATAL(node->get_logger(), "%s refused the thresholds: %s", kCollisionService,
                 response->error.c_str());
    return false;
  }

  RCLCPP_INFO(node->get_logger(),
              "collision thresholds set — force upper (%.0f, %.0f, %.0f) N / (%.0f, %.0f, %.0f) Nm",
              upper_force[0], upper_force[1], upper_force[2], upper_force[3], upper_force[4],
              upper_force[5]);
  return true;
}

CartesianImpedanceController::Gains CartesianImpedanceController::readGainParameters() {
  auto node = get_node();
  Gains gains;
  gains.stiffness = diagonalGain(node->get_parameter("translational_stiffness").as_double(),
                                 node->get_parameter("rotational_stiffness").as_double());
  gains.damping = diagonalGain(node->get_parameter("translational_damping").as_double(),
                               node->get_parameter("rotational_damping").as_double());
  gains.ki = diagonalGain(node->get_parameter("translational_ki").as_double(),
                          node->get_parameter("rotational_ki").as_double());

  const double t_clip = std::abs(node->get_parameter("translational_clip").as_double());
  const double r_clip = std::abs(node->get_parameter("rotational_clip").as_double());
  gains.clip.translation_min = Eigen::Vector3d::Constant(-t_clip);
  gains.clip.translation_max = Eigen::Vector3d::Constant(t_clip);
  gains.clip.rotation_min = Eigen::Vector3d::Constant(-r_clip);
  gains.clip.rotation_max = Eigen::Vector3d::Constant(r_clip);

  gains.nullspace_stiffness = node->get_parameter("nullspace_stiffness").as_double();
  gains.joint1_nullspace_stiffness = node->get_parameter("joint1_nullspace_stiffness").as_double();
  // Floored, not just validated: scheduleWaypoint divides by these, so zero puts the trajectory's
  // last knot at infinity and the arm holds forever with nothing logged. They are live-tunable, so
  // a bad value can arrive long after startup.
  gains.max_pos_speed = clampedSpeed(node, "max_pos_speed");
  gains.max_rot_speed = clampedSpeed(node, "max_rot_speed");
  return gains;
}

CallbackReturn CartesianImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  auto node = get_node();
  arm_id_ = node->get_parameter("arm_id").as_string();
  joint_names_.clear();
  for (int i = 1; i <= kNumJoints; ++i) {
    joint_names_.push_back(arm_id_ + "_joint" + std::to_string(i));
  }
  base_frame_ = node->get_parameter("base_frame").as_string();
  tcp_frame_ = node->get_parameter("tcp_frame").as_string();
  gains_.writeFromNonRT(readGainParameters());

  // Poll rather than hook add_on_set_parameters_callback: that callback fires BEFORE the store is
  // updated, so it would have to reconstruct the new value from the incoming vector by name — ~30
  // lines of string matching to save 0.5 s of latency on a knob a human turns.
  // ponytail: 2 Hz poll of ~12 doubles; switch to a post-set callback if rclcpp ever gains one here.
  gain_refresh_timer_ = node->create_wall_timer(std::chrono::milliseconds(500), [this]() {
    gains_.writeFromNonRT(readGainParameters());
  });

  robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      arm_id_ + "/" + kRobotModelInterface, arm_id_ + "/" + kRobotStateInterface);

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  target_sub_ = node->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
      node->get_parameter("target_topic").as_string(), rclcpp::QoS(10),
      [this](const trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg) {
        onTrajectory(msg);
      });

  collision_client_ =
      node->create_client<franka_msgs::srv::SetFullCollisionBehavior>(kCollisionService);
  // Hard failure, not a warning. The alternative is activating a torque controller against unknown
  // reflex thresholds, which is invisible in TF, the topic list and this controller's own logs —
  // a session where this silently did not apply looks exactly like one where the gains are wrong.
  if (!applyCollisionBehavior()) {
    RCLCPP_FATAL(node->get_logger(),
                 "Refusing to configure without known collision thresholds. See "
                 "docs/franka-inference-bringup.md, \"Collision thresholds\".");
    return CallbackReturn::ERROR;
  }

  const Gains& startup = *gains_.readFromNonRT();
  RCLCPP_INFO(node->get_logger(),
              "cartesian impedance: K=(%.0f N/m, %.0f Nm/rad) D=(%.0f, %.0f) clip=(%.3f m, %.3f "
              "rad) -> max force ~%.0f N (all tunable live via set_parameters)",
              startup.stiffness(0, 0), startup.stiffness(3, 3), startup.damping(0, 0),
              startup.damping(3, 3), startup.clip.translation_max(0), startup.clip.rotation_max(0),
              startup.stiffness(0, 0) * startup.clip.translation_max(0));
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  gains_.writeFromNonRT(readGainParameters());

  position_interfaces_.clear();
  velocity_interfaces_.clear();
  if (!controller_interface::get_ordered_interfaces(state_interfaces_, joint_names_,
                                                    hardware_interface::HW_IF_POSITION,
                                                    position_interfaces_) ||
      !controller_interface::get_ordered_interfaces(state_interfaces_, joint_names_,
                                                    hardware_interface::HW_IF_VELOCITY,
                                                    velocity_interfaces_)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "Could not resolve position/velocity state interfaces for all 7 joints of '%s'.",
                 arm_id_.c_str());
    return CallbackReturn::ERROR;
  }

  // polyumi_tcp relative to franka's O_T_EE frame. Both are fixed frames, so one lookup holds for
  // the controller's lifetime, and taking it from TF means nuc/tcp_calib.py remains the only place
  // the geometry is written down.
  const std::string ee_frame = arm_id_ + "_hand_tcp";
  try {
    const auto tf = tf_buffer_->lookupTransform(ee_frame, tcp_frame_, tf2::TimePointZero,
                                                tf2::durationFromSec(5.0));
    ee_to_tcp_rotation_ = Eigen::Quaterniond(tf.transform.rotation.w, tf.transform.rotation.x,
                                             tf.transform.rotation.y, tf.transform.rotation.z);
    ee_to_tcp_ = Eigen::Vector3d(tf.transform.translation.x, tf.transform.translation.y,
                                 tf.transform.translation.z);
  } catch (const tf2::TransformException& e) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "No transform %s -> %s after 5s: %s. fr3_bringup publishes it; without it the "
                 "control point would silently be the wrist, ~15 cm from the fingertips.",
                 ee_frame.c_str(), tcp_frame_.c_str(), e.what());
    return CallbackReturn::ERROR;
  }

  Pose pose;
  Jacobian jacobian;
  Vector7d q;
  Vector7d dq;
  readState(pose, jacobian, q, dq);

  // Seed the equilibrium at where the arm already is, so the first update() sees zero error and
  // commands nothing. Any other seed kicks the arm the instant this controller activates.
  interpolator_.writeFromNonRT(
      std::make_shared<const PoseTrajectoryInterpolator>(get_node()->now().seconds(), pose));
  last_waypoint_time_ = get_node()->now().seconds();
  q_nullspace_ = q;
  integral_error_.setZero();
  tau_previous_.setZero();
  active_.store(true, std::memory_order_release);

  RCLCPP_INFO(get_node()->get_logger(),
              "activated holding %s at (%.3f, %.3f, %.3f); %s -> %s offset (%.4f, %.4f, %.4f)",
              tcp_frame_.c_str(), pose.position.x(), pose.position.y(), pose.position.z(),
              ee_frame.c_str(), tcp_frame_.c_str(), ee_to_tcp_.x(), ee_to_tcp_.y(),
              ee_to_tcp_.z());
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  active_.store(false, std::memory_order_release);
  robot_model_->release_interfaces();
  position_interfaces_.clear();
  velocity_interfaces_.clear();
  return CallbackReturn::SUCCESS;
}

// ---------------------------------------------------------------------------
// Target chunks (non-realtime)
// ---------------------------------------------------------------------------

void CartesianImpedanceController::onTrajectory(
    const trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg) {
  // Only update() advances now_seconds_, so splicing while inactive would schedule against a
  // frozen clock. on_activate re-seeds the interpolator anyway, so nothing is lost by dropping.
  if (!active_.load(std::memory_order_acquire)) {
    return;
  }
  if (!msg->header.frame_id.empty() && msg->header.frame_id != base_frame_) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                         "Ignoring chunk in frame '%s'; this controller commands in '%s'.",
                         msg->header.frame_id.c_str(), base_frame_.c_str());
    return;
  }

  const double anchor = rclcpp::Time(msg->header.stamp).seconds();
  const double curr_time = now_seconds_.load(std::memory_order_relaxed);
  const Gains& speeds = *gains_.readFromNonRT();

  auto interp = *interpolator_.readFromNonRT();
  if (!interp) {
    return;
  }

  int spliced = 0;
  for (const auto& point : msg->points) {
    if (point.transforms.empty()) {
      continue;
    }
    const auto& tf = point.transforms.front();
    Pose pose;
    pose.position = Eigen::Vector3d(tf.translation.x, tf.translation.y, tf.translation.z);
    pose.orientation =
        Eigen::Quaterniond(tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z);
    pose.orientation.normalize();

    const double target_time = anchor + rclcpp::Duration(point.time_from_start).seconds();
    // scheduleWaypoint ignores anything at or before curr_time; skipping here keeps the
    // bookkeeping honest about how much of the chunk was actually usable.
    if (target_time <= curr_time) {
      continue;
    }

    interp = std::make_shared<const PoseTrajectoryInterpolator>(interp->scheduleWaypoint(
        pose, target_time, curr_time, last_waypoint_time_, speeds.max_pos_speed,
        speeds.max_rot_speed));
    last_waypoint_time_ = target_time;
    ++spliced;
  }

  if (spliced == 0) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                         "Whole chunk of %zu waypoints was already in the past; arm is holding. "
                         "Check latency.arm_exec and the laptop/NUC clock sync.",
                         msg->points.size());
    return;
  }
  interpolator_.writeFromNonRT(interp);
}

// ---------------------------------------------------------------------------
// Realtime loop
// ---------------------------------------------------------------------------

void CartesianImpedanceController::readState(Pose& pose,
                                             Jacobian& jacobian,
                                             Vector7d& q,
                                             Vector7d& dq) {
  for (int i = 0; i < kNumJoints; ++i) {
    q(i) = position_interfaces_[i].get().get_value();
    dq(i) = velocity_interfaces_[i].get().get_value();
  }

  const std::array<double, 16> ee_pose = robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  const Pose ee = poseFromColumnMajor(ee_pose);

  // Move both the pose and the Jacobian onto polyumi_tcp. The offset must be rotated into the base
  // frame first: ee_to_tcp_ is expressed in the EE frame, the Jacobian is not.
  const Eigen::Vector3d offset_base = ee.orientation * ee_to_tcp_;
  pose.position = ee.position + offset_base;
  pose.orientation = ee.orientation * ee_to_tcp_rotation_;

  const std::array<double, 42> jacobian_array =
      robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  jacobian = shiftJacobian(Eigen::Map<const Jacobian>(jacobian_array.data()), offset_base);
}

controller_interface::return_type CartesianImpedanceController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& /*period*/) {
  const double now = time.seconds();
  now_seconds_.store(now, std::memory_order_relaxed);

  Pose pose;
  Jacobian jacobian;
  Vector7d q;
  Vector7d dq;
  readState(pose, jacobian, q, dq);

  const auto& interp = *interpolator_.readFromRT();
  if (!interp) {
    return controller_interface::return_type::OK;
  }
  // Past the last waypoint the interpolator clamps, so a stalled action stream becomes a position
  // hold rather than an extrapolation off the end of the workspace.
  const Pose desired = (*interp)(now);

  const Gains& gains = *gains_.readFromRT();
  const Vector6d error = clipPoseError(poseError(pose, desired), gains.clip);
  integral_error_ = accumulateIntegralError(integral_error_, error);

  const Vector7d coriolis =
      Eigen::Map<const Vector7d>(robot_model_->getCoriolisForceVector().data());
  const Vector7d tau =
      taskTorque(jacobian, error, integral_error_, dq, gains.stiffness, gains.damping, gains.ki) +
      nullspaceTorque(jacobian, q, dq, q_nullspace_, gains.nullspace_stiffness,
                      gains.joint1_nullspace_stiffness) +
      coriolis;

  tau_previous_ = saturateTorqueRate(tau, tau_previous_, kMaxTorqueRate);
  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_.at(i).set_value(tau_previous_(i));
  }

  return controller_interface::return_type::OK;
}

}  // namespace polyumi_fr3_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(polyumi_fr3_controllers::CartesianImpedanceController,
                       controller_interface::ControllerInterface)

// Copyright (c) 2026 PolyUMI. MIT.
//
// Drives the Franka Hand from the gripper half of the inference action chunk, holding it as a
// horizon of absolutely-timed widths and planning one Move at a time against a measured model of
// what a Move costs.
//
// WHY LIBFRANKA DIRECTLY, NOT franka_gripper. A Move cannot be pre-empted: stop() queues *behind*
// it and costs the remainder of the stroke plus ~100 ms, and a superseding Move stops the fingers
// rather than redirecting them. So the only sane discipline is run-to-completion, and owning the
// connection makes that structural -- exactly one thread calls move(), in a loop, so there is no
// busy flag, no outstanding-goal bookkeeping and no way to have two moves in flight. Through the
// action server it would be a convention instead, and franka_gripper accepts every goal into a
// detached thread with no queue, so nothing would enforce it.
//
// WHAT IT COSTS. blockedDuration(0) = 363 ms, so the command ceiling is 2.75 Hz and realistically
// 0.7-1.7 Hz. Against a 10 Hz setpoint stream this node is a DECIMATOR, servicing roughly every
// 4th-15th setpoint. That is the hardware, not a bug. See docs/crb-fr3-inference.md for what the
// hand cannot do.
//
// It also republishes /fr3_gripper/joint_states: with load_gripper:=false nothing else does, and
// six consumers need it (policy_client_node's agent_pos[7] among them). That stream costs the
// libfranka connection, which only one process may hold, so it is tied to `execute`. With
// execute:=false the node neither connects nor publishes state, and an arm-only session can run
// with the hand unplugged; policy_client_node then falls back to the closed width unless
// require_gripper_state is set.

#include <polyumi_fr3_controllers/gripper_trajectory_interpolator.hpp>

#include <franka/exception.h>
#include <franka/gripper.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using polyumi_fr3_controllers::blockedDuration;
using polyumi_fr3_controllers::HandLimits;
using polyumi_fr3_controllers::MoveCommand;
using polyumi_fr3_controllers::selectMove;
using polyumi_fr3_controllers::WidthTrajectory;

/// How long the mover waits for a chunk when it has nothing to do. Short against the hand's own
/// 363 ms floor, so it costs nothing; it exists only so a stopped stream does not spin.
constexpr auto kIdleWait = std::chrono::milliseconds(20);

/// Backoff after a failed read. A read that fails returns immediately rather than waiting for a
/// datagram, so without this a dropped connection spins a core. Far below the hand's ~180 ms state
/// interval, so a recovered connection loses nothing.
constexpr auto kReadRetryWait = std::chrono::milliseconds(20);

/// Divergence between where a successful Move said the fingers are and what they report, past
/// which the "it never got there" warning fires.
constexpr double kDivergenceWarn = 0.005;

class FrankaHandNode : public rclcpp::Node {
 public:
  FrankaHandNode() : rclcpp::Node("fr3_gripper") {
    execute_ = declare_parameter("execute", false);
    const auto robot_ip = declare_parameter("robot_ip", std::string("192.168.51.20"));
    const auto target_topic = declare_parameter("target_topic", std::string("/polyumi/target_gripper"));
    const auto state_topic = declare_parameter("state_topic", std::string("~/joint_states"));
    // 0.0 means "ask the hand", which is better than the guessed constant the launch file used to
    // have to supply. Anything else is a backstop clamp.
    max_width_ = declare_parameter("max_width_m", 0.0);
    // A time budget, not a position tolerance: the smallest stroke worth 0.6 s of deafness.
    deadband_ = declare_parameter("width_deadband_m", 0.005);
    min_speed_ = declare_parameter("min_speed_mps", 0.005);
    horizon_s_ = declare_parameter("horizon_s", 3.0);
    // Homing sweeps the full stroke, so it is opt-in.
    const bool home_on_start = declare_parameter("home_on_start", false);

    limits_.v_max = declare_parameter("v_max_mps", limits_.v_max);
    limits_.a_max = declare_parameter("a_max_mps2", limits_.a_max);
    limits_.cmd_delay = declare_parameter("cmd_delay_s", limits_.cmd_delay);
    limits_.fixed_cost = declare_parameter("fixed_cost_s", limits_.fixed_cost);
    limits_.t_obs_delay = declare_parameter("t_obs_delay_s", limits_.t_obs_delay);
    validateParams();

    state_pub_ = create_publisher<sensor_msgs::msg::JointState>(state_topic, 10);
    target_sub_ = create_subscription<trajectory_msgs::msg::JointTrajectory>(
        target_topic, 10, [this](trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
          onTarget(*msg);
        });

    // Connecting is what CLAIMS the hand -- only one process may hold it -- so it is gated on
    // execute rather than done unconditionally. An arm-only session therefore needs neither a
    // reachable nor a homed hand, and no state stream is published (see the file header).
    if (execute_) {
      gripper_ = std::make_unique<franka::Gripper>(robot_ip);
      if (home_on_start) {
        RCLCPP_INFO(get_logger(), "Homing (sweeps the full stroke)...");
        gripper_->homing();
      }
      const franka::GripperState state = gripper_->readOnce();
      if (state.max_width <= 0.0) {
        // An unhomed hand reports max_width 0 and move() returns TRUE while doing nothing at all.
        // Refusing here is the only way that failure is ever visible. The connection stays open so
        // the state stream keeps running; it is only execution that is refused.
        RCLCPP_ERROR(get_logger(),
                     "Hand reports max_width=0: it is NOT HOMED. Refusing to execute. Relaunch with "
                     "home_on_start:=true, or home it from Desk.");
        execute_ = false;
      }
      if (max_width_ <= 0.0) {
        max_width_ = state.max_width;
      }
      // Write the resolved width back, so the PARAMETER reports the clamp actually in force rather
      // than the 0.0 "ask the hand" sentinel. gripper_range_probe reads it to tell a physical stop
      // from a software one, and 0.0 makes every open endpoint look like a clamp.
      set_parameter(rclcpp::Parameter("max_width_m", max_width_));
      reader_ = std::thread(&FrankaHandNode::readLoop, this);
    }
    RCLCPP_INFO(get_logger(), "Franka Hand %s: max_width %.4f m, execute=%s",
                gripper_ ? "connected" : "NOT CONNECTED (no /fr3_gripper/joint_states)", max_width_,
                execute_ ? "TRUE (MOVES THE FINGERS)" : "false (plans and logs only)");

    mover_ = std::thread(&FrankaHandNode::moveLoop, this);
  }

  ~FrankaHandNode() override {
    running_ = false;
    cv_.notify_all();
    if (reader_.joinable()) {
      reader_.join();
    }
    if (mover_.joinable()) {
      mover_.join();
    }
  }

 private:
  void validateParams() {
    const auto bad = [](const std::string& why) {
      throw std::invalid_argument("Invalid franka_hand_node configuration: " + why);
    };
    if (limits_.v_max <= 0.0 || limits_.a_max <= 0.0) {
      bad("v_max_mps and a_max_mps2 must be positive");
    }
    if (deadband_ < 0.0 || min_speed_ <= 0.0 || horizon_s_ <= 0.0) {
      bad("width_deadband_m must be >= 0, min_speed_mps and horizon_s > 0");
    }
    // selectMove clamps the planned speed into [min_speed, v_max], and std::clamp is undefined
    // with the bounds inverted.
    if (min_speed_ > limits_.v_max) {
      bad("min_speed_mps must not exceed v_max_mps");
    }
    // t_obs_delay is subtracted from cmd_delay to get the scheduling delay, and a Move can never
    // start moving after it has already returned.
    if (!(limits_.t_obs_delay < limits_.cmd_delay && limits_.cmd_delay <= limits_.fixed_cost)) {
      bad("need t_obs_delay_s < cmd_delay_s <= fixed_cost_s");
    }
  }

  /// Free-running state stream. Proven by csv_gripper_timing_probe to coexist with a blocking
  /// move() on the same Gripper, which is what lets the mover thread block without going blind.
  void readLoop() {
    while (running_) {
      try {
        const franka::GripperState state = gripper_->readOnce();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          measured_ = state.width;
        }
        sensor_msgs::msg::JointState msg;
        msg.header.stamp = now();
        msg.name = {"fr3_finger_joint1", "fr3_finger_joint2"};
        // Each finger carries half the aperture; consumers sum them. Same contract as
        // franka_gripper, which is why nothing downstream had to change.
        msg.position = {state.width / 2.0, state.width / 2.0};
        state_pub_->publish(msg);
      } catch (const franka::Exception& e) {
        // Reads legitimately fail while a command is in flight. Not fatal, and not rare.
        RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 1000, "gripper read: %s", e.what());
        std::this_thread::sleep_for(kReadRetryWait);
      }
    }
  }

  /// Absolute-time setpoints spliced into the horizon in place. header.stamp + time_from_start is
  /// a real schedule, not a shape -- see policy_client_node's _actions_to_gripper_trajectory.
  void onTarget(const trajectory_msgs::msg::JointTrajectory& msg) {
    const double stamp = rclcpp::Time(msg.header.stamp).seconds();
    std::vector<double> times;
    std::vector<double> widths;
    for (const auto& pt : msg.points) {
      if (pt.positions.empty()) {
        continue;
      }
      times.push_back(stamp + rclcpp::Duration(pt.time_from_start).seconds());
      // Upper bound only once the stroke is known: disconnected, max_width_ is the unresolved 0.0
      // sentinel and clamping to it would flatten every width to zero.
      const double w = std::max(pt.positions[0], 0.0);
      widths.push_back(max_width_ > 0.0 ? std::min(w, max_width_) : w);
    }
    if (times.empty()) {
      return;
    }

    const double t = now().seconds();
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      horizon_ = horizon_.splice(WidthTrajectory(std::move(times), std::move(widths)), t)
                     .prune(t, horizon_s_);
    } catch (const std::invalid_argument& e) {
      // Unsorted times from a mis-built chunk. Drop it rather than taking the node down.
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "bad gripper chunk: %s", e.what());
      return;
    }
    // Bumped under no lock ordering constraint, but bumped BEFORE the notify: a chunk that arrives
    // while the mover is inside move() must leave the predicate already true, or the wakeup is lost
    // and the node stalls until the next chunk.
    ++chunk_seq_;
    cv_.notify_one();
  }

  void moveLoop() {
    std::uint64_t seen = 0;
    while (running_) {
      std::optional<MoveCommand> cmd;
      std::optional<double> x;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        const double t = now().seconds();
        horizon_ = horizon_.prune(t, horizon_s_);
        x = estimate();
        cmd = selectMove(horizon_, x, t, deadband_, min_speed_, limits_);
      }

      if (!cmd.has_value()) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, kIdleWait, [&] { return !running_ || chunk_seq_ != seen; });
        seen = chunk_seq_;
        continue;
      }
      seen = chunk_seq_;

      const double predicted =
          blockedDuration(std::abs(cmd->width - x.value_or(cmd->width)), cmd->speed, limits_);
      RCLCPP_INFO(get_logger(), "move(%.4f m, %.4f m/s) %s, ~%.0f ms%s", cmd->width, cmd->speed,
                  cmd->on_time ? "on time" : "CHASING", predicted * 1e3,
                  execute_ ? "" : " [dry run]");

      if (!execute_) {
        // Sleep the predicted block so the dry run reproduces the real command cadence, which is
        // the whole point of watching it before letting it move anything. Disconnected there is no
        // measured width to plan from, so every decision comes off selectMove's no-state branch:
        // the cadence is still real, the targets are open-loop.
        std::this_thread::sleep_for(std::chrono::duration<double>(predicted));
        continue;
      }
      runMove(*cmd);
    }
  }

  void runMove(const MoveCommand& cmd) {
    bool ok = false;
    try {
      ok = gripper_->move(cmd.width, cmd.speed);
    } catch (const franka::Exception& e) {
      RCLCPP_WARN(get_logger(), "move failed: %s", e.what());
    }

    std::lock_guard<std::mutex> lock(mutex_);
    // A Move that returned true leaves the fingers AT the commanded width and stationary, and
    // nothing can change that for another fixed_cost. That beats the width field, which refreshes
    // at ~5.5 Hz -- 180 ms, or 8 mm of travel, out of date.
    //
    // Unless the two disagree by more than that staleness could explain, which is the case the
    // warning below names. Then the commanded width is a fiction the planner would keep working
    // from, comparing it against the deadband and concluding it has already arrived. A false
    // return means declined or stalled. Either way the measurement is the only truth there is.
    const bool diverged =
        measured_.has_value() && std::abs(*measured_ - cmd.width) > kDivergenceWarn;
    commanded_ = (ok && !diverged) ? std::optional<double>(cmd.width) : std::nullopt;
    if (ok && diverged) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Move reported success at %.4f m but the hand reads %.4f m -- it may be "
                           "stalled on an object (Move applies no force; use Grasp to hold). "
                           "Planning from the measurement instead.",
                           cmd.width, *measured_);
    }
  }

  /// Best available finger position. Caller must hold mutex_.
  std::optional<double> estimate() const { return commanded_.has_value() ? commanded_ : measured_; }

  std::unique_ptr<franka::Gripper> gripper_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr target_sub_;

  HandLimits limits_;
  bool execute_ = false;
  double max_width_ = 0.0;
  double deadband_ = 0.005;
  double min_speed_ = 0.005;
  double horizon_s_ = 3.0;

  // ponytail: plain mutex, no RT path here; revisit only if the reader misses datagrams.
  // Not realtime_tools::RealtimeBuffer -- that is single-reader by contract and there are two.
  std::mutex mutex_;
  std::condition_variable cv_;
  WidthTrajectory horizon_;
  std::optional<double> measured_;
  std::optional<double> commanded_;
  std::uint64_t chunk_seq_ = 0;

  std::atomic<bool> running_{true};
  std::thread reader_;
  std::thread mover_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrankaHandNode>());
  rclcpp::shutdown();
  return 0;
}

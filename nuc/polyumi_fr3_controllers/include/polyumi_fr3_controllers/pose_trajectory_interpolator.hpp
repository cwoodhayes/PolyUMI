// Copyright (c) 2026 PolyUMI. MIT.
//
// C++/Eigen port of UMI's umi/common/pose_trajectory_interpolator.py, which is what turns a
// sparse action chunk into the continuous reference a 1 kHz servo needs.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <utility>
#include <vector>

namespace polyumi_fr3_controllers {

/// A pose on the trajectory: position plus orientation, both in the trajectory's own frame.
struct Pose {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

/// Translational distance (m) and angular distance (rad) between two poses.
std::pair<double, double> poseDistance(const Pose& a, const Pose& b);

/**
 * Piecewise-linear position / slerp orientation interpolation over absolutely-timed waypoints.
 *
 * Ported from UMI (upstream `PoseTrajectoryInterpolator`). The value here is not the interpolation
 * itself but `scheduleWaypoint`: it splices a future waypoint into a trajectory that is already
 * being consumed, without a discontinuity at the current instant. That is what lets action chunks
 * arriving at 10 Hz drive a 1 kHz loop that never stops between them.
 *
 * Evaluation is allocation-free and safe to call from a realtime update(). Splicing is not — build
 * the new interpolator off the realtime thread and hand it over.
 *
 * Times are absolute seconds on whatever clock the caller uses; this class never reads a clock.
 */
class PoseTrajectoryInterpolator {
 public:
  /// Constant trajectory: evaluates to `pose` at every instant.
  PoseTrajectoryInterpolator(double time, const Pose& pose);

  /// Trajectory through `poses` at `times`, which must be non-empty, equal in length, and sorted.
  PoseTrajectoryInterpolator(std::vector<double> times, std::vector<Pose> poses);

  /// Interpolate at `t`, clamped to the trajectory's own endpoints (never extrapolates).
  Pose operator()(double t) const;

  double firstTime() const { return times_.front(); }
  double lastTime() const { return times_.back(); }
  std::size_t size() const { return times_.size(); }

  /// The sub-trajectory spanning [start_t, end_t], with both endpoints materialised as waypoints.
  PoseTrajectoryInterpolator trim(double start_t, double end_t) const;

  /**
   * Splice `pose` in as a waypoint at absolute instant `time`.
   *
   * Everything after `last_waypoint_time` is discarded, so a fresh action chunk supersedes the
   * tail of the previous one while the part already being executed is left alone. A waypoint at or
   * before `curr_time` is ignored — it refers to an instant that has passed.
   *
   * `max_pos_speed` / `max_rot_speed` stretch the segment's duration if reaching the waypoint on
   * time would demand more than that, which bounds how far the reference can run away from the
   * arm. UMI leaves both infinite; we do not, because that distance is what sets contact force.
   */
  PoseTrajectoryInterpolator scheduleWaypoint(const Pose& pose,
                                              double time,
                                              double curr_time,
                                              double last_waypoint_time,
                                              double max_pos_speed,
                                              double max_rot_speed) const;

 private:
  std::vector<double> times_;
  std::vector<Pose> poses_;
};

}  // namespace polyumi_fr3_controllers

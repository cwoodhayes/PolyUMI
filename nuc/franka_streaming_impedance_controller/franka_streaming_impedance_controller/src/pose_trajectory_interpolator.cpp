// Copyright (c) 2026 the franka_streaming_impedance_controller authors. MIT.

#include <franka_streaming_impedance_controller/pose_trajectory_interpolator.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace franka_streaming_impedance {

std::pair<double, double> poseDistance(const Pose& a, const Pose& b) {
  return {(b.position - a.position).norm(), a.orientation.angularDistance(b.orientation)};
}

PoseTrajectoryInterpolator::PoseTrajectoryInterpolator(double time, const Pose& pose)
    : times_{time}, poses_{pose} {}

PoseTrajectoryInterpolator::PoseTrajectoryInterpolator(std::vector<double> times,
                                                       std::vector<Pose> poses)
    : times_(std::move(times)), poses_(std::move(poses)) {
  if (times_.empty() || times_.size() != poses_.size()) {
    throw std::invalid_argument("PoseTrajectoryInterpolator: need equal, non-empty times and poses");
  }
  if (!std::is_sorted(times_.begin(), times_.end())) {
    throw std::invalid_argument("PoseTrajectoryInterpolator: times must be non-decreasing");
  }
}

Pose PoseTrajectoryInterpolator::operator()(double t) const {
  t = std::clamp(t, times_.front(), times_.back());

  // First waypoint at or after t; the segment we want ends there.
  const auto upper = std::lower_bound(times_.begin(), times_.end(), t);
  if (upper == times_.begin()) {
    return poses_.front();
  }
  const std::size_t hi = static_cast<std::size_t>(upper - times_.begin());
  if (hi >= times_.size()) {
    return poses_.back();
  }
  const std::size_t lo = hi - 1;

  const double span = times_[hi] - times_[lo];
  // Coincident waypoints carry no direction; take the later one rather than dividing by zero.
  const double alpha = (span > 0.0) ? (t - times_[lo]) / span : 1.0;

  Pose out;
  out.position = poses_[lo].position + alpha * (poses_[hi].position - poses_[lo].position);
  out.orientation = poses_[lo].orientation.slerp(alpha, poses_[hi].orientation);
  return out;
}

PoseTrajectoryInterpolator PoseTrajectoryInterpolator::trim(double start_t, double end_t) const {
  std::vector<double> times;
  std::vector<Pose> poses;
  times.reserve(times_.size() + 2);
  poses.reserve(times_.size() + 2);

  // start_t and end_t become real waypoints; interior ones are kept, and the strict inequalities
  // are what stop either endpoint being added twice.
  const auto push = [&](double t) {
    if (times.empty() || t > times.back()) {
      times.push_back(t);
      poses.push_back((*this)(t));
    }
  };

  push(start_t);
  for (const double t : times_) {
    if (t > start_t && t < end_t) {
      push(t);
    }
  }
  push(end_t);

  return PoseTrajectoryInterpolator(std::move(times), std::move(poses));
}

PoseTrajectoryInterpolator PoseTrajectoryInterpolator::scheduleWaypoint(const Pose& pose,
                                                                       double time,
                                                                       double curr_time,
                                                                       double last_waypoint_time,
                                                                       double max_pos_speed,
                                                                       double max_rot_speed) const {
  // The waypoint refers to an instant that has already passed.
  if (time <= curr_time) {
    return *this;
  }

  // Window of the existing trajectory to keep. Ported from UMI verbatim: the sequence of min/max
  // operations is what establishes start_time <= end_time <= time and curr_time <= start_time,
  // which everything below relies on. Rearranging it is how you get a discontinuity at curr_time.
  double start_time = std::max(curr_time, times_.front());
  double end_time = (time <= last_waypoint_time) ? curr_time
                                                 : std::max(last_waypoint_time, curr_time);
  end_time = std::min(end_time, time);
  start_time = std::min(start_time, end_time);

  PoseTrajectoryInterpolator trimmed = trim(start_time, end_time);

  // Stretch the segment if arriving on time would exceed either speed limit.
  const Pose end_pose = trimmed(end_time);
  const auto [pos_dist, rot_dist] = poseDistance(pose, end_pose);
  const double duration = std::max({time - end_time, pos_dist / max_pos_speed,
                                    rot_dist / max_rot_speed});
  const double new_waypoint_time = end_time + duration;

  std::vector<double> times = trimmed.times_;
  std::vector<Pose> poses = trimmed.poses_;
  // Belt and braces, and currently unreachable: trim() always ends exactly at end_time, and
  // duration is at least `time - end_time` which is strictly positive, so new_waypoint_time is
  // always past the last knot. Kept because the constructor THROWS on non-increasing times and
  // this runs in a subscription callback — if the min/max sequence above is ever rearranged, this
  // is what turns a crash into a dropped point.
  while (!times.empty() && times.back() >= new_waypoint_time) {
    times.pop_back();
    poses.pop_back();
  }
  times.push_back(new_waypoint_time);
  poses.push_back(pose);

  return PoseTrajectoryInterpolator(std::move(times), std::move(poses));
}

}  // namespace franka_streaming_impedance

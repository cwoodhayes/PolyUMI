// Copyright (c) 2026 PolyUMI. MIT.

#include <polyumi_fr3_controllers/gripper_trajectory_interpolator.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace polyumi_fr3_controllers {

namespace {

/// Drop setpoints whose instant has passed -- but never the last one.
///
/// The terminal setpoint is where the signal is HEADING, and stays true long after the instant it
/// was due; an intermediate one does not. Keeping it means a chunk that lands entirely late -- a
/// single "be here now" point, or any chunk under clock skew -- still moves the fingers, via
/// selectMove's chase branch, rather than freezing them.
void dropElapsed(std::vector<double>& times, std::vector<double>& widths, double curr_time) {
  std::size_t keep = 0;
  while (keep + 1 < times.size() && times[keep] <= curr_time) {
    ++keep;
  }
  times.erase(times.begin(), times.begin() + static_cast<std::ptrdiff_t>(keep));
  widths.erase(widths.begin(), widths.begin() + static_cast<std::ptrdiff_t>(keep));
}

}  // namespace

double moveDuration(double dx, double speed, const HandLimits& limits) {
  dx = std::abs(dx);
  if (dx <= 0.0) {
    return 0.0;
  }
  // The hand clips silently rather than refusing, so a command above v_max behaves as v_max.
  const double v = std::min(speed, limits.v_max);
  if (v <= 0.0) {
    // A Move at zero speed never completes. Say so rather than dividing by it.
    return std::numeric_limits<double>::infinity();
  }
  if (dx >= v * v / limits.a_max) {
    return dx / v + v / limits.a_max;  // trapezoidal
  }
  return 2.0 * std::sqrt(dx / limits.a_max);  // triangular: v is never reached, so it drops out
}

double blockedDuration(double dx, double speed, const HandLimits& limits) {
  return limits.fixed_cost + moveDuration(dx, speed, limits);
}

double maxDistance(double tau, const HandLimits& limits) {
  if (tau <= 0.0) {
    return 0.0;
  }
  // Below 2*v_max/a there is no cruise phase at all -- the ramp up meets the ramp down.
  if (tau < 2.0 * limits.v_max / limits.a_max) {
    return limits.a_max * tau * tau / 4.0;
  }
  return limits.v_max * (tau - limits.v_max / limits.a_max);
}

std::optional<double> speedForDuration(double dx, double tau, const HandLimits& limits) {
  dx = std::abs(dx);
  if (dx <= 0.0) {
    return 0.0;
  }
  // Relative slack, because the caller's dx and maxDistance's are computed by different routes and
  // land an ulp apart at the exact boundary -- and a move the hand can just barely make is the
  // wrong one to reject.
  if (dx > maxDistance(tau, limits) * (1.0 + 1e-9)) {
    return std::nullopt;
  }

  // dx/v + v/a = tau has two roots; the smaller is the slow one, which is the one we want. The
  // feasibility check above makes the discriminant non-negative bar that same ulp.
  const double disc = std::sqrt(std::max(0.0, tau * tau - 4.0 * dx / limits.a_max));
  // Rationalised form. The textbook (a/2)*(tau - disc) is algebraically identical and cancels
  // catastrophically for a small dx over a long tau -- which is the common case here, so it is
  // wrong in exactly the regime that matters. There is a test pinning this.
  return 2.0 * dx / (tau + disc);
}

WidthTrajectory::WidthTrajectory(std::vector<double> times, std::vector<double> widths)
    : times_(std::move(times)), widths_(std::move(widths)) {
  if (times_.empty() || times_.size() != widths_.size()) {
    throw std::invalid_argument("WidthTrajectory: need equal, non-empty times and widths");
  }
  if (!std::is_sorted(times_.begin(), times_.end())) {
    throw std::invalid_argument("WidthTrajectory: times must be non-decreasing");
  }
}

WidthTrajectory WidthTrajectory::splice(const WidthTrajectory& chunk, double curr_time) const {
  // Everything at or after the chunk's first instant is superseded by it. dropElapsed then trims
  // what the splice leaves, so a chunk entirely in the past narrows to its own final setpoint.
  const double cut = chunk.empty() ? std::numeric_limits<double>::infinity() : chunk.firstTime();

  std::vector<double> times;
  std::vector<double> widths;
  const auto push = [&](double t, double w) {
    times.push_back(t);
    widths.push_back(w);
  };

  for (std::size_t i = 0; i < times_.size(); ++i) {
    if (times_[i] < cut) {
      push(times_[i], widths_[i]);
    }
  }
  for (std::size_t i = 0; i < chunk.size(); ++i) {
    push(chunk.times_[i], chunk.widths_[i]);
  }
  dropElapsed(times, widths, curr_time);

  if (times.empty()) {
    return WidthTrajectory();
  }
  return WidthTrajectory(std::move(times), std::move(widths));
}

WidthTrajectory WidthTrajectory::prune(double curr_time, double horizon) const {
  std::vector<double> times;
  std::vector<double> widths;
  for (std::size_t i = 0; i < times_.size(); ++i) {
    if (times_[i] <= curr_time + horizon) {
      times.push_back(times_[i]);
      widths.push_back(widths_[i]);
    }
  }
  dropElapsed(times, widths, curr_time);

  if (times.empty()) {
    return WidthTrajectory();
  }
  return WidthTrajectory(std::move(times), std::move(widths));
}

double WidthTrajectory::operator()(double t) const {
  if (times_.empty()) {
    throw std::invalid_argument("WidthTrajectory: cannot evaluate an empty trajectory");
  }
  t = std::clamp(t, times_.front(), times_.back());

  const auto upper = std::lower_bound(times_.begin(), times_.end(), t);
  if (upper == times_.begin()) {
    return widths_.front();
  }
  const std::size_t hi = static_cast<std::size_t>(upper - times_.begin());
  if (hi >= times_.size()) {
    return widths_.back();
  }
  const std::size_t lo = hi - 1;

  const double span = times_[hi] - times_[lo];
  // Coincident setpoints carry no slope; take the later one rather than dividing by zero.
  const double alpha = (span > 0.0) ? (t - times_[lo]) / span : 1.0;
  return widths_[lo] + alpha * (widths_[hi] - widths_[lo]);
}

std::optional<MoveCommand> selectMove(const WidthTrajectory& horizon,
                                      std::optional<double> width_now,
                                      double curr_time,
                                      double deadband,
                                      double min_speed,
                                      const HandLimits& limits) {
  if (horizon.empty()) {
    return std::nullopt;
  }
  // Nothing measured yet: go to the first setpoint flat out and let the next call plan properly.
  if (!width_now.has_value()) {
    return MoveCommand{horizon.firstWidth(), limits.v_max, horizon.firstTime(), false};
  }
  const double x = *width_now;

  // The earliest instant the fingers can be anywhere. Everything before it is already unreachable
  // no matter what we command, so it is the origin for every feasibility question below.
  const double t0 = curr_time + limits.commandDelay();

  // (A) The earliest setpoint we can still reach on time. Earliest rather than best: it commits
  // the least and lets the next chunk re-plan soonest.
  for (std::size_t i = 0; i < horizon.size(); ++i) {
    const double t_i = horizon.times()[i];
    if (t_i <= t0) {
      continue;
    }
    const double dx = std::abs(horizon.widths()[i] - x);
    if (dx < deadband) {
      continue;  // a fixed_cost of deafness to go nowhere
    }
    const auto v = speedForDuration(dx, t_i - t0, limits);
    if (v.has_value()) {
      return MoveCommand{horizon.widths()[i], std::clamp(*v, min_speed, limits.v_max), t_i, true};
    }
  }

  // (B) Nothing is reachable on time, so chase: run flat out at where the signal will be by the
  // time we arrive. Stopping short at an intermediate width that lands on time is strictly worse,
  // because covering the remainder afterwards costs another whole fixed_cost.
  double w = horizon.lastWidth();
  double arrival = horizon.lastTime();
  for (int i = 0; i < 3; ++i) {  // fixed point; converges in two on any realistic horizon
    arrival = t0 + moveDuration(std::abs(w - x), limits.v_max, limits);
    w = horizon(arrival);  // clamped past the horizon end
  }
  if (std::abs(w - x) < deadband) {
    return std::nullopt;
  }
  return MoveCommand{w, limits.v_max, arrival, false};
}

}  // namespace polyumi_fr3_controllers

// Copyright (c) 2026 PolyUMI. MIT.
//
// The Franka Hand's answer to PoseTrajectoryInterpolator, and a very different problem.
//
// The arm takes a continuous reference at 1 kHz. The hand takes discrete, blocking, un-preemptable
// Move commands at under 2 Hz, so there is nothing to interpolate *to* — the question is instead
// which setpoint each Move should aim at, and how fast. That needs a model of what a Move costs,
// which is what the first half of this header is. Its constants were fitted in a Jupyter notebook
// from runs of the franka_hand_testing probes (see that directory); the notebook is not in the repo.

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace polyumi_fr3_controllers {

/**
 * Measured response of the Franka Hand to a Move.
 *
 * Every field is a knob rather than a constant because `fixed_cost` drifts 50-70 ms between runs,
 * and because `t_obs_delay` is a guess nobody can currently measure.
 *
 * Note which constants are contaminated by `t_obs_delay` and which are not, because it decides
 * where the correction is applied. `v_max`, `a_max` and `fixed_cost` were all timed off move()'s
 * blocking duration -- both endpoints inside our own process, microseconds, no width sample
 * anywhere -- so they are clean. `cmd_delay` was recovered from the width trace, so it is
 * `physical delay + t_obs_delay` and must have the observation lag taken back out before it is
 * used to schedule anything.
 */
struct HandLimits {
  /// Speed ceiling. The hand ACCEPTS higher and silently clips; it never refuses, so nothing
  /// downstream will raise an error if this is exceeded.
  double v_max = 0.1153;  // m/s
  /// Acceleration. Sets the triangular/trapezoidal crossover at v_max^2/a_max = 37 mm, below which
  /// the commanded speed has no effect at all.
  double a_max = 0.360;  // m/s^2
  /// Send -> fingers start moving, as OBSERVED in the width stream. Constant over a 20x speed range.
  double cmd_delay = 0.208;  // s
  /// Send -> move() returns, at zero travel. The floor under every command: even a 0 mm move costs
  /// this, which is why a deadband is mandatory and why the command ceiling is 1/0.363 = 2.75 Hz.
  double fixed_cost = 0.363;  // s
  /// How far the reported width lags physical reality. UNMEASURABLE without a camera on the
  /// fingers; this is a guess. It shifts commands later so the fingers themselves land on time,
  /// rather than their reported state landing on time.
  double t_obs_delay = 0.050;  // s

  /// Send -> fingers move, with the observation lag removed: the delay that actually schedules.
  double commandDelay() const { return cmd_delay - t_obs_delay; }
};

/// Time the fingers spend moving `dx` when commanded at `speed`. Trapezoidal, or triangular when
/// the stroke is too short to reach `speed`. `speed` is clipped to `v_max` internally.
double moveDuration(double dx, double speed, const HandLimits& limits = {});

/// How long move() blocks: `moveDuration` plus the fixed cost. `blockedDuration(0, v)` is the floor.
double blockedDuration(double dx, double speed, const HandLimits& limits = {});

/// Farthest the fingers can travel in `tau` seconds of motion. Zero for `tau <= 0`.
double maxDistance(double tau, const HandLimits& limits = {});

/**
 * Slowest speed that covers `dx` in `tau` seconds of motion, or nullopt if `tau` is too short.
 *
 * Arriving exactly on time beats arriving early and waiting, which is what makes this the inverse
 * worth having rather than just commanding v_max everywhere.
 */
std::optional<double> speedForDuration(double dx, double tau, const HandLimits& limits = {});

/**
 * Absolutely-timed width setpoints, the scalar analogue of PoseTrajectoryInterpolator.
 *
 * Same contract: value semantics, never reads a clock, clamps rather than extrapolating, and the
 * constructor rejects ragged or unsorted input. No speed limiting here, unlike the arm's version --
 * the hand's limits bound the COMMAND, not the reference, so they live in selectMove.
 */
class WidthTrajectory {
 public:
  /// Empty trajectory: nothing scheduled. This is the idle state, not an error.
  WidthTrajectory() = default;

  /// Setpoints `widths` at `times`, which must be equal in length and sorted.
  WidthTrajectory(std::vector<double> times, std::vector<double> widths);

  /**
   * Splice `chunk` in, superseding the overlapping tail.
   *
   * Everything at or after the chunk's first instant is discarded and replaced; everything before
   * it survives; everything at or before `curr_time` is dropped as elapsed -- EXCEPT the very last
   * point, which survives regardless of its instant. That exception is what lets a wholly-past
   * chunk still command something: the terminal setpoint is where the signal is heading, so a
   * fully elapsed splice narrows to a single point rather than emptying out.
   */
  WidthTrajectory splice(const WidthTrajectory& chunk, double curr_time) const;

  /// Drop elapsed setpoints and anything past `curr_time + horizon`, bounding a rogue publisher.
  /// Same never-drop-the-last-point exception as splice().
  WidthTrajectory prune(double curr_time, double horizon) const;

  /// Linear interpolation at `t`, clamped to the endpoints. Throws if empty.
  double operator()(double t) const;

  bool empty() const { return times_.empty(); }
  std::size_t size() const { return times_.size(); }
  const std::vector<double>& times() const { return times_; }
  const std::vector<double>& widths() const { return widths_; }
  double firstTime() const { return times_.front(); }
  double lastTime() const { return times_.back(); }
  double firstWidth() const { return widths_.front(); }
  double lastWidth() const { return widths_.back(); }

 private:
  std::vector<double> times_;
  std::vector<double> widths_;
};

/// A Move to issue, as chosen by selectMove.
struct MoveCommand {
  double width;        ///< target to command
  double speed;        ///< speed to command
  double target_time;  ///< the setpoint instant being aimed at; diagnostics only
  bool on_time;        ///< false means the chase branch: we know we will arrive late
};

/**
 * Choose the next Move, or nothing.
 *
 * Two branches. The first walks the horizon for the EARLIEST setpoint still reachable in time and
 * aims at it with the slowest speed that lands on schedule -- earliest, not best, because it
 * commits least and re-plans soonest. When nothing is reachable the second branch chases: it
 * commands full speed toward wherever the trajectory will be by the time the fingers arrive.
 *
 * Chasing rather than stopping short is deliberate. Commanding an intermediate width that arrives
 * exactly on time is strictly worse, because covering the remainder afterwards costs another whole
 * `fixed_cost`.
 *
 * `width_now` is nullopt before the first state has arrived.
 */
std::optional<MoveCommand> selectMove(const WidthTrajectory& horizon,
                                      std::optional<double> width_now,
                                      double curr_time,
                                      double deadband,
                                      double min_speed,
                                      const HandLimits& limits = {});

}  // namespace polyumi_fr3_controllers

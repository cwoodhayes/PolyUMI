// Copyright (c) 2026 PolyUMI. MIT.
//
// The model constants under test were measured on the hand and derived in
// notebooks/gripper_free_running.ipynb, which is the source of truth for all of them. The anchor
// tests at the bottom of the model section replay rows straight out of notebooks/data/, so editing
// a constant fails here rather than quietly changing what the fingers do.

#include <polyumi_fr3_controllers/gripper_trajectory_interpolator.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

using polyumi_fr3_controllers::blockedDuration;
using polyumi_fr3_controllers::HandLimits;
using polyumi_fr3_controllers::maxDistance;
using polyumi_fr3_controllers::moveDuration;
using polyumi_fr3_controllers::selectMove;
using polyumi_fr3_controllers::speedForDuration;
using polyumi_fr3_controllers::WidthTrajectory;

namespace {

constexpr double kTol = 1e-9;
const HandLimits kL{};

/// Stroke at which the trapezoid degenerates into a triangle at full speed: 37 mm.
double crossover(double v, const HandLimits& l = kL) { return v * v / l.a_max; }

}  // namespace

// ---------------------------------------------------------------------------- the response model

TEST(GripperModel, TrapezoidAboveCrossover) {
  const double dx = 0.060;  // well past the 37 mm crossover
  EXPECT_NEAR(moveDuration(dx, 0.100), 0.060 / 0.100 + 0.100 / kL.a_max, kTol);
}

TEST(GripperModel, TriangleBelowCrossover) {
  const double dx = 0.002;
  EXPECT_NEAR(moveDuration(dx, 0.100), 2.0 * std::sqrt(dx / kL.a_max), kTol);
}

TEST(GripperModel, BelowCrossoverTheCommandedSpeedDoesNothing) {
  // The fingers never reach the commanded speed, so it drops out of the answer entirely. This is
  // why fine policy corrections cannot be made faster by asking.
  const double dx = 0.002;
  EXPECT_NEAR(moveDuration(dx, 0.030), moveDuration(dx, 0.100), kTol);
}

TEST(GripperModel, BranchesAgreeAtTheCrossover) {
  const double v = 0.080;
  const double dx = crossover(v);
  EXPECT_NEAR(dx / v + v / kL.a_max, 2.0 * std::sqrt(dx / kL.a_max), kTol);
  EXPECT_NEAR(moveDuration(dx, v), 2.0 * std::sqrt(dx / kL.a_max), kTol);
}

TEST(GripperModel, SpeedAboveVMaxClipsRatherThanRefusing) {
  // The hand accepts any speed and silently delivers v_max, so nothing downstream ever errors.
  EXPECT_NEAR(moveDuration(0.060, 10.0), moveDuration(0.060, kL.v_max), kTol);
}

TEST(GripperModel, ZeroSpeedNeverCompletes) {
  EXPECT_TRUE(std::isinf(moveDuration(0.060, 0.0)));
}

TEST(GripperModel, BlockedDurationAtZeroStrokeIsTheFixedCost) {
  // The 363 ms floor: this is the entire justification for the deadband, and for the 2.75 Hz
  // ceiling on how often the node can command anything at all.
  EXPECT_NEAR(blockedDuration(0.0, kL.v_max), kL.fixed_cost, kTol);
}

TEST(GripperModel, DirectionDoesNotMatter) {
  EXPECT_NEAR(moveDuration(-0.040, 0.050), moveDuration(0.040, 0.050), kTol);
}

TEST(GripperModel, InverseRoundTripsInBothBranches) {
  for (const double dx : {0.002, 0.020, 0.037, 0.060, 0.082}) {
    for (const double v : {0.010, 0.050, 0.100}) {
      const double tau = moveDuration(dx, v);
      const auto back = speedForDuration(dx, tau);
      ASSERT_TRUE(back.has_value()) << "dx=" << dx << " v=" << v;
      EXPECT_NEAR(moveDuration(dx, *back), tau, kTol) << "dx=" << dx << " v=" << v;
    }
  }
}

TEST(GripperModel, InverseRejectsDurationsShorterThanTheTriangle) {
  const double dx = 0.060;
  const double floor_tau = moveDuration(dx, kL.v_max);
  EXPECT_TRUE(speedForDuration(dx, floor_tau * 0.99).has_value() == false);
  EXPECT_TRUE(speedForDuration(dx, floor_tau).has_value());
}

TEST(GripperModel, SpeedForDurationIsStableForTinyDistances) {
  // The regime the textbook (a/2)*(tau - sqrt(tau^2 - 4dx/a)) form destroys: the two terms agree to
  // ten digits and their difference is the whole answer. This is the COMMON case here -- a
  // millimetre-scale correction scheduled a second out -- not a corner.
  const double dx = 1e-9;
  const double tau = 5.0;
  const auto v = speedForDuration(dx, tau);
  ASSERT_TRUE(v.has_value());
  EXPECT_NEAR(moveDuration(dx, *v), tau, kTol);
}

TEST(GripperModel, MaxDistanceIsZeroForNonPositiveTau) {
  EXPECT_NEAR(maxDistance(0.0), 0.0, kTol);
  EXPECT_NEAR(maxDistance(-1.0), 0.0, kTol);
}

TEST(GripperModel, MaxDistanceIsExactlyWhatTheInverseWillStillAccept) {
  for (const double tau : {0.2, 0.6406, 1.0, 3.0}) {
    const double dmax = maxDistance(tau);
    EXPECT_TRUE(speedForDuration(dmax, tau).has_value()) << "tau=" << tau;
    EXPECT_FALSE(speedForDuration(dmax * 1.01, tau).has_value()) << "tau=" << tau;
  }
}

TEST(GripperModel, MatchesTheMeasuredStrokeSweep) {
  // Two rows from notebooks/data/stroke_commands.csv: a 60 mm stroke at 100 and at 30 mm/s, each
  // the mean of three reps. Per-move jitter is ~50 ms, so 100 ms is the honest tolerance.
  EXPECT_NEAR(blockedDuration(0.060, 0.100), 1.2645, 0.1);
  EXPECT_NEAR(blockedDuration(0.060, 0.030), 2.4612, 0.1);
}

// ------------------------------------------------------------------------------- t_obs_delay

TEST(GripperObsDelay, ZeroDelayLeavesTheMeasuredCommandDelayAlone) {
  HandLimits l;
  l.t_obs_delay = 0.0;
  EXPECT_NEAR(l.commandDelay(), l.cmd_delay, kTol);
}

TEST(GripperObsDelay, AssumingObservationLagCommandsASlowerMove) {
  // The reported width lags reality, so the fingers can start sooner than the trace suggests. That
  // buys a larger budget for the same setpoint, and the slowest on-time speed drops accordingly --
  // which is what makes the FINGERS land on time rather than their reported state.
  const WidthTrajectory h({1.0}, {0.040});
  HandLimits lagged;             // t_obs_delay = 0.050
  HandLimits raw;
  raw.t_obs_delay = 0.0;

  const auto a = selectMove(h, 0.020, 0.0, 0.005, 0.001, lagged);
  const auto b = selectMove(h, 0.020, 0.0, 0.005, 0.001, raw);
  ASSERT_TRUE(a.has_value() && b.has_value());
  EXPECT_LT(a->speed, b->speed);
}

// ----------------------------------------------------------------------------- the horizon buffer

TEST(WidthTrajectoryTest, RejectsRaggedAndUnsortedInput) {
  EXPECT_THROW(WidthTrajectory({0.0, 1.0}, {0.01}), std::invalid_argument);
  EXPECT_THROW(WidthTrajectory({}, {}), std::invalid_argument);
  EXPECT_THROW(WidthTrajectory({1.0, 0.0}, {0.01, 0.02}), std::invalid_argument);
}

TEST(WidthTrajectoryTest, InterpolatesLinearly) {
  const WidthTrajectory h({0.0, 1.0}, {0.010, 0.030});
  EXPECT_NEAR(h(0.25), 0.015, kTol);
}

TEST(WidthTrajectoryTest, ClampsRatherThanExtrapolating) {
  const WidthTrajectory h({1.0, 2.0}, {0.010, 0.030});
  EXPECT_NEAR(h(-5.0), 0.010, kTol);
  EXPECT_NEAR(h(99.0), 0.030, kTol);
}

TEST(WidthTrajectoryTest, EvaluatingWhenEmptyThrows) {
  EXPECT_THROW(WidthTrajectory()(0.0), std::invalid_argument);
}

TEST(WidthTrajectoryTest, SpliceReplacesTheOverlappingTailAndKeepsThePrefix) {
  const WidthTrajectory h({1.0, 2.0, 3.0}, {0.010, 0.020, 0.030});
  const WidthTrajectory chunk({2.0, 4.0}, {0.070, 0.080});
  const WidthTrajectory out = h.splice(chunk, 0.0);

  ASSERT_EQ(out.size(), 3u);
  EXPECT_NEAR(out.times()[0], 1.0, kTol);
  EXPECT_NEAR(out.widths()[0], 0.010, kTol);  // prefix survives
  EXPECT_NEAR(out.widths()[1], 0.070, kTol);  // t=2 superseded, t=3 gone
  EXPECT_NEAR(out.widths()[2], 0.080, kTol);
}

TEST(WidthTrajectoryTest, SpliceDropsElapsedSetpoints) {
  const WidthTrajectory h({1.0, 2.0}, {0.010, 0.020});
  const WidthTrajectory out = h.splice(WidthTrajectory({3.0}, {0.030}), 1.5);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_NEAR(out.firstTime(), 2.0, kTol);
}

TEST(WidthTrajectoryTest, AnEntirelyPastChunkLeavesNothing) {
  // The clock-skew case. Empty is the idle state, not an error: the fingers hold position.
  const WidthTrajectory h({1.0}, {0.010});
  EXPECT_TRUE(h.splice(WidthTrajectory({0.5}, {0.030}), 10.0).empty());
}

TEST(WidthTrajectoryTest, PruneBoundsBothEnds) {
  const WidthTrajectory h({1.0, 2.0, 3.0, 9.0}, {0.010, 0.020, 0.030, 0.040});
  const WidthTrajectory out = h.prune(1.5, 3.0);  // keep (1.5, 4.5]
  ASSERT_EQ(out.size(), 2u);
  EXPECT_NEAR(out.firstTime(), 2.0, kTol);
  EXPECT_NEAR(out.lastTime(), 3.0, kTol);
}

// ---------------------------------------------------------------------------------- selection

TEST(SelectMove, EmptyHorizonCommandsNothing) {
  EXPECT_FALSE(selectMove(WidthTrajectory(), 0.020, 0.0, 0.005, 0.001).has_value());
}

TEST(SelectMove, WithNoStateEstimateItGoesToTheFirstSetpointFlatOut) {
  const WidthTrajectory h({1.0, 2.0}, {0.030, 0.060});
  const auto cmd = selectMove(h, std::nullopt, 0.0, 0.005, 0.001);
  ASSERT_TRUE(cmd.has_value());
  EXPECT_NEAR(cmd->width, 0.030, kTol);
  EXPECT_NEAR(cmd->speed, kL.v_max, kTol);
  EXPECT_FALSE(cmd->on_time);
}

TEST(SelectMove, AReachableSetpointGetsTheSlowestOnTimeSpeed) {
  // Not v_max: arriving early and sitting there is worse than arriving on the instant asked for.
  const WidthTrajectory h({2.0}, {0.040});
  const auto cmd = selectMove(h, 0.020, 0.0, 0.005, 0.001);
  ASSERT_TRUE(cmd.has_value());
  EXPECT_TRUE(cmd->on_time);
  EXPECT_NEAR(cmd->width, 0.040, kTol);
  EXPECT_LT(cmd->speed, kL.v_max);
  EXPECT_NEAR(moveDuration(0.020, cmd->speed), 2.0 - kL.commandDelay(), kTol);
}

TEST(SelectMove, SpeedIsClampedToMinSpeed) {
  // A very distant setpoint asks for a speed the hand would take forever to finish at; min_speed
  // is the floor that keeps the move from outliving the horizon that scheduled it.
  const WidthTrajectory h({20.0}, {0.040});
  const auto cmd = selectMove(h, 0.020, 0.0, 0.005, 0.005);
  ASSERT_TRUE(cmd.has_value());
  EXPECT_NEAR(cmd->speed, 0.005, kTol);
}

TEST(SelectMove, DeadbandSetpointsAreSkipped) {
  // 2 mm is not worth 0.4 s of being unable to react to anything.
  const WidthTrajectory h({2.0}, {0.022});
  EXPECT_FALSE(selectMove(h, 0.020, 0.0, 0.005, 0.001).has_value());
}

TEST(SelectMove, TheEarliestFeasibleSetpointWins) {
  const WidthTrajectory h({1.0, 2.0}, {0.030, 0.060});
  const auto cmd = selectMove(h, 0.020, 0.0, 0.005, 0.001);
  ASSERT_TRUE(cmd.has_value());
  EXPECT_NEAR(cmd->width, 0.030, kTol);
}

TEST(SelectMove, SetpointsInsideTheCommandDelayAreNeverClaimedOnTime) {
  const WidthTrajectory h({0.05}, {0.010});  // sooner than commandDelay() = 0.158 s
  const auto cmd = selectMove(h, 0.080, 0.0, 0.005, 0.001);
  ASSERT_TRUE(cmd.has_value());
  EXPECT_FALSE(cmd->on_time);
}

TEST(SelectMove, AnUnreachableRampChasesThePredictedArrivalNotTheFirstMissedPoint) {
  // A 70 mm collapse in 300 ms is physically impossible. Aiming at the first missed setpoint
  // (60 mm) wastes a whole fixed_cost covering the remainder afterwards; chasing to where the
  // signal will be by the time we arrive gets there in one move.
  const WidthTrajectory h({0.2, 0.3, 0.4, 0.5}, {0.060, 0.040, 0.020, 0.010});
  const auto cmd = selectMove(h, 0.080, 0.0, 0.005, 0.001);
  ASSERT_TRUE(cmd.has_value());
  EXPECT_FALSE(cmd->on_time);
  EXPECT_NEAR(cmd->width, 0.010, kTol);
  EXPECT_NEAR(cmd->speed, kL.v_max, kTol);
}

TEST(SelectMove, ChasingStopsWhenAlreadyInsideTheDeadband) {
  const WidthTrajectory h({0.01, 0.02}, {0.0205, 0.0215});
  EXPECT_FALSE(selectMove(h, 0.020, 0.0, 0.005, 0.001).has_value());
}

TEST(SelectMove, CloseThenHoldThenOpenCommandsTheClose) {
  // THE design test. From open, the policy wants: close now, hold closed for 1.2 s, then open.
  // "Aim at the last horizon point" never closes the gripper and silently breaks every
  // pick-and-place. "Aim at the earliest missed point" gets this right only by accident.
  const WidthTrajectory h({0.1, 0.5, 0.9, 1.3, 1.4, 1.5},
                          {0.010, 0.010, 0.010, 0.010, 0.010, 0.080});
  const auto cmd = selectMove(h, 0.080, 0.0, 0.005, 0.001);
  ASSERT_TRUE(cmd.has_value());
  EXPECT_NEAR(cmd->width, 0.010, kTol);
  EXPECT_TRUE(cmd->on_time);
}

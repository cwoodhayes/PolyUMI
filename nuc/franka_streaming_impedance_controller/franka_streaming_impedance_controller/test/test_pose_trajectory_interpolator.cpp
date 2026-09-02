// Copyright (c) 2026 the franka_streaming_impedance_controller authors. MIT.
//
// Every expected value here was produced by RUNNING UMI's own implementation
// (../universal_manipulation_interface/umi/common/pose_trajectory_interpolator.py) on the same
// inputs and pasting what it printed. That is the point: a port checked against itself only
// proves it is self-consistent. See the plan's verification section for the generator snippet.

#include <franka_streaming_impedance_controller/pose_trajectory_interpolator.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using franka_streaming_impedance::Pose;
using franka_streaming_impedance::PoseTrajectoryInterpolator;
using franka_streaming_impedance::poseDistance;

namespace {

constexpr double kTol = 1e-9;
constexpr double kInf = std::numeric_limits<double>::infinity();

Pose makePose(double x, double y, double z, double yaw) {
  Pose p;
  p.position = Eigen::Vector3d(x, y, z);
  p.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  return p;
}

void expectPose(const Pose& got, double x, double y, double z, double qz, double qw) {
  EXPECT_NEAR(got.position.x(), x, kTol);
  EXPECT_NEAR(got.position.y(), y, kTol);
  EXPECT_NEAR(got.position.z(), z, kTol);
  // Compare on the shorter arc: q and -q are the same rotation.
  const double sign = (got.orientation.w() < 0.0) ? -1.0 : 1.0;
  EXPECT_NEAR(sign * got.orientation.z(), qz, kTol);
  EXPECT_NEAR(sign * got.orientation.w(), qw, kTol);
  EXPECT_NEAR(got.orientation.x(), 0.0, kTol);
  EXPECT_NEAR(got.orientation.y(), 0.0, kTol);
}

// The fixture UMI's numbers were generated from: (0,0,0)@t=10 -> (1,2,3) yawed 90deg @t=11.
PoseTrajectoryInterpolator twoWaypoints() {
  return PoseTrajectoryInterpolator({10.0, 11.0},
                                    {makePose(0, 0, 0, 0), makePose(1, 2, 3, M_PI / 2)});
}

}  // namespace

TEST(PoseTrajectoryInterpolator, ConstantTrajectoryHoldsItsSeed) {
  // The activation case: seeded at the arm's current pose, it must command that pose at every
  // instant until a chunk arrives. Anything else moves the arm the moment the controller starts.
  const PoseTrajectoryInterpolator interp(10.0, makePose(0.3, -0.1, 0.5, 0.25));

  for (const double t : {0.0, 10.0, 1e6}) {
    expectPose(interp(t), 0.3, -0.1, 0.5, std::sin(0.125), std::cos(0.125));
  }
}

TEST(PoseTrajectoryInterpolator, MatchesUmiLerpAndSlerp) {
  const auto interp = twoWaypoints();

  expectPose(interp(10.0), 0.0, 0.0, 0.0, 0.0, 1.0);
  expectPose(interp(10.25), 0.25, 0.5, 0.75, 0.195090322016, 0.980785280403);
  expectPose(interp(10.5), 0.5, 1.0, 1.5, 0.382683432365, 0.923879532511);
  expectPose(interp(11.0), 1.0, 2.0, 3.0, 0.707106781187, 0.707106781187);
}

TEST(PoseTrajectoryInterpolator, ClampsRatherThanExtrapolating) {
  // Running off either end must hold the endpoint. Extrapolating past the last waypoint is how a
  // stalled action stream turns into the arm accelerating away from the workspace.
  const auto interp = twoWaypoints();

  expectPose(interp(9.5), 0.0, 0.0, 0.0, 0.0, 1.0);
  expectPose(interp(11.5), 1.0, 2.0, 3.0, 0.707106781187, 0.707106781187);
}

TEST(PoseTrajectoryInterpolator, ScheduleWaypointSplicesWithoutDiscontinuity) {
  // The whole reason this class exists. Splicing at curr_time=10.5 must leave the trajectory
  // evaluating to exactly what it did there a moment ago; a jump is a jerk on real hardware.
  const auto interp = twoWaypoints();
  const Pose before = interp(10.5);

  const auto spliced = interp.scheduleWaypoint(makePose(5, 0, 0, 0), 12.0, 10.5, 11.0, kInf, kInf);

  ASSERT_EQ(spliced.size(), 3u);
  EXPECT_NEAR(spliced.firstTime(), 10.5, kTol);
  EXPECT_NEAR(spliced.lastTime(), 12.0, kTol);

  const Pose after = spliced(10.5);
  EXPECT_NEAR((after.position - before.position).norm(), 0.0, kTol);
  EXPECT_NEAR(after.orientation.angularDistance(before.orientation), 0.0, kTol);

  // UMI's values for the spliced trajectory.
  expectPose(spliced(11.0), 1.0, 2.0, 3.0, 0.707106781187, 0.707106781187);
  expectPose(spliced(11.5), 3.0, 1.0, 1.5, 0.382683432365, 0.923879532511);
  expectPose(spliced(12.0), 5.0, 0.0, 0.0, 0.0, 1.0);
}

TEST(PoseTrajectoryInterpolator, WaypointInThePastIsIgnored) {
  // Chunks cross a network and arrive late. A waypoint whose instant has passed must not become
  // the arm's new target, or every delayed chunk yanks it backwards.
  const auto interp = twoWaypoints();

  const auto same = interp.scheduleWaypoint(makePose(9, 9, 9, 0), 10.4, 10.5, 11.0, kInf, kInf);

  ASSERT_EQ(same.size(), 2u);
  EXPECT_NEAR(same.firstTime(), 10.0, kTol);
  EXPECT_NEAR(same.lastTime(), 11.0, kTol);
}

TEST(PoseTrajectoryInterpolator, SpeedLimitStretchesTheSegment) {
  // 10 m away at 1 m/s cannot be reached in the 0.5 s requested, so the waypoint moves out to
  // t=21.0 instead of the reference sprinting there. This is the bound on how far the equilibrium
  // point can lead the arm, which is the bound on impedance force.
  const auto interp = twoWaypoints();

  const auto limited =
      interp.scheduleWaypoint(makePose(11, 2, 3, M_PI / 2), 11.5, 10.5, 11.0, 1.0, kInf);

  ASSERT_EQ(limited.size(), 3u);
  EXPECT_NEAR(limited.lastTime(), 21.0, kTol);
  expectPose(limited(21.0), 11.0, 2.0, 3.0, 0.707106781187, 0.707106781187);
}

TEST(PoseTrajectoryInterpolator, PoseDistanceSplitsTranslationFromRotation) {
  const auto [pos, rot] = poseDistance(makePose(0, 0, 0, 0), makePose(3, 4, 0, M_PI / 2));

  EXPECT_NEAR(pos, 5.0, kTol);
  EXPECT_NEAR(rot, M_PI / 2, kTol);
}

// The speed limit is ours, not UMI's — upstream splices with max_pos_speed = max_rot_speed = inf.
// That divergence breaks an invariant UMI relies on: its caller stores the REQUESTED waypoint time
// in `last_waypoint_time` (franka_interpolation_controller.py:351), which upstream is always the
// trajectory's tail because nothing ever stretches a segment. Once a limit can stretch one, the
// caller's record and the real tail are different numbers, and `scheduleWaypoint` uses the record
// for two things at once: detecting a rewind, and bounding the trim window.
//
// These two tests pin what that divergence actually costs, because it is not obvious from reading:
// the path keeps its shape, and the speed bound holds, but the reference's LAG is unbounded.

TEST(PoseTrajectoryInterpolator, SpeedLimitedChunkKeepsItsPathShape) {
  // A right-angle chunk: the first waypoint is 1.5 m out along +x, unreachable in the 0.5 s asked
  // for at 1 m/s, so the limiter engages and stays engaged; the rest of the chunk turns and runs
  // along +y. The corner is what discriminates. Feeding the trajectory's real tail back in as
  // `last_waypoint_time` instead of the requested time sends every following waypoint down the
  // rewind branch, which collapses the whole chunk to one straight segment from the origin to the
  // last pose — the arm would cut the corner. Here the first leg must still be a straight run
  // along +x with y pinned at zero.
  const double t0 = 0.0;
  PoseTrajectoryInterpolator interp(t0, makePose(0, 0, 0, 0));
  double last_waypoint = t0;

  for (int k = 0; k < 6; ++k) {
    const double target = t0 + 0.5 + 0.1 * k;
    interp = interp.scheduleWaypoint(makePose(1.5, 0.05 * k, 0, 0), target, t0, last_waypoint, 1.0,
                                     kInf);
    last_waypoint = target;
  }

  // The leading leg is pure +x: the limiter stretched it, it was not discarded.
  EXPECT_NEAR(interp(0.25).position.x(), 0.25, 1e-3);
  EXPECT_NEAR(interp(0.25).position.y(), 0.0, 1e-6);
  EXPECT_NEAR(interp(0.50).position.x(), 0.50, 1e-3);
  EXPECT_NEAR(interp(0.50).position.y(), 0.0, 1e-6);

  // It still arrives at the chunk's final pose, and holds there.
  EXPECT_NEAR(interp(interp.lastTime()).position.x(), 1.5, kTol);
  EXPECT_NEAR(interp(interp.lastTime()).position.y(), 0.25, kTol);
}

TEST(PoseTrajectoryInterpolator, SpeedLimitBoundsReferenceSpeedButNotItsLag) {
  // What the limit does and does not buy. It caps how fast the equilibrium point may TRAVEL, which
  // is what bounds contact force. It does NOT cap how far behind the policy the reference may fall:
  // a stream commanding 1.5 m/s against a 1.0 m/s limit accumulates the 0.5 m/s deficit forever, so
  // the trajectory's tail runs ever further into the future and the arm keeps crawling toward a
  // stale target long after the policy has stopped.
  //
  // Pinned deliberately rather than left to be discovered on the arm. If a cap on how far the tail
  // may lead `now` is ever added, this is the test that must change, and changing it should be a
  // decision rather than a surprise.
  constexpr double kMaxSpeed = 1.0;
  double curr = 0.0;
  double last_waypoint = 0.0;
  PoseTrajectoryInterpolator interp(curr, makePose(0, 0, 0, 0));

  for (int chunk = 0; chunk < 40; ++chunk) {
    curr = 0.3 * chunk;
    for (int k = 0; k < 16; ++k) {
      const double target = curr + 0.1 * (k + 1);
      interp = interp.scheduleWaypoint(makePose(0.15 * (chunk * 3 + k + 1), 0, 0, 0), target, curr,
                                       last_waypoint, kMaxSpeed, kInf);
      last_waypoint = target;
    }
  }

  // The reference never outruns the limit, anywhere on the trajectory.
  for (double t = curr; t + 0.1 <= interp.lastTime(); t += 0.1) {
    const double travelled = (interp(t + 0.1).position - interp(t).position).norm();
    EXPECT_LE(travelled, kMaxSpeed * 0.1 + 1e-9) << "reference exceeded max_pos_speed at t=" << t;
  }

  // Knots stay bounded — the trim window discards what `curr` has passed, so a stream that limits
  // forever does not grow the trajectory forever.
  EXPECT_LE(interp.size(), 20u);

  // But the backlog does grow without bound: ~0.5 s of deficit per second of streaming.
  EXPECT_GT(interp.lastTime() - curr, 5.0) << "expected the reference to fall behind; if this now "
                                              "fails, a lead cap was added and the comment above "
                                              "is stale";
}

TEST(PoseTrajectoryInterpolator, RepeatedSplicesDoNotGrowTheTrajectory) {
  // A 10 Hz chunk stream splices continuously for the length of an episode, so every splice runs in
  // a subscription callback that also frees the previous trajectory. The trim window is what keeps
  // that bounded: it discards whatever curr_time has passed, so the knot count settles instead of
  // tracking the episode's length.
  //
  // Coincident knots — what scipy raises on in UMI — are not checked here because they cannot
  // arise: trim's interior filter is strict and the new waypoint's duration is always positive, so
  // the knots are strictly increasing by construction. See the note on the pop-back guard in
  // scheduleWaypoint.
  auto interp = twoWaypoints();

  for (int i = 0; i < 200; ++i) {
    const double now = 10.5 + 0.1 * i;
    interp = interp.scheduleWaypoint(makePose(0.01 * i, 0, 0, 0), now + 0.3, now, now + 0.2,
                                     kInf, kInf);
    ASSERT_LE(interp.size(), 8u) << "spliced trajectory is growing without bound";
  }
  EXPECT_GT(interp.size(), 1u) << "test is vacuous if the splices are not landing";
}

// Copyright (c) 2026 PolyUMI. MIT.
//
// Every expected value here was produced by RUNNING UMI's own implementation
// (../universal_manipulation_interface/umi/common/pose_trajectory_interpolator.py) on the same
// inputs and pasting what it printed. That is the point: a port checked against itself only
// proves it is self-consistent. See the plan's verification section for the generator snippet.

#include <polyumi_fr3_controllers/pose_trajectory_interpolator.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using polyumi_fr3_controllers::Pose;
using polyumi_fr3_controllers::PoseTrajectoryInterpolator;
using polyumi_fr3_controllers::poseDistance;

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

TEST(PoseTrajectoryInterpolator, RepeatedSplicesStayStrictlyIncreasing) {
  // A 10 Hz chunk stream splices continuously for the length of an episode. Coincident knots are
  // what scipy raises on in UMI; here they must simply be superseded, forever, without growing.
  auto interp = twoWaypoints();

  for (int i = 0; i < 200; ++i) {
    const double now = 10.5 + 0.1 * i;
    interp = interp.scheduleWaypoint(makePose(0.01 * i, 0, 0, 0), now + 0.3, now, now + 0.2,
                                     kInf, kInf);
    for (std::size_t k = 1; k < interp.size(); ++k) {
      ASSERT_GT(interp.trim(interp.firstTime(), interp.lastTime()).size(), 0u);
    }
    ASSERT_LE(interp.size(), 8u) << "spliced trajectory is growing without bound";
  }
}

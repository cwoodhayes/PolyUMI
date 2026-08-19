// Copyright (c) 2026 PolyUMI. MIT.
//
// The law runs at 1 kHz on a 3 kg arm with no firmware net underneath it, so its properties are
// pinned here rather than discovered in the room. Everything below uses a hand-built Jacobian and
// needs no hardware.

#include <polyumi_fr3_controllers/cartesian_impedance_law.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace polyumi_fr3_controllers;  // NOLINT(build/namespaces)

namespace {

constexpr double kTol = 1e-9;

Pose makePose(double x, double y, double z, double yaw = 0.0) {
  Pose p;
  p.position = Eigen::Vector3d(x, y, z);
  p.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  return p;
}

/// A full-rank, well-conditioned 6x7 Jacobian. The exact numbers do not matter; being full rank
/// with a real 1-D null space does.
Jacobian wellConditionedJacobian() {
  Jacobian j = Jacobian::Zero();
  for (int i = 0; i < 6; ++i) {
    j(i, i) = 1.0;
  }
  j(0, 6) = 0.5;
  j(3, 6) = -0.25;
  return j;
}

/// Recover the end-effector force a joint torque corresponds to, via tau = J^T f.
/// Uses an UNDAMPED inverse deliberately: the law's own pseudo-inverse is damped (lambda=0.2), and
/// inverting with it would silently fold that bias into every expected force here.
Eigen::VectorXd forceFromTorque(const Jacobian& jacobian, const Vector7d& tau) {
  return dampedPseudoInverse(jacobian.transpose(), 0.0) * tau;
}

Matrix6d diagonal(double translational, double rotational) {
  Matrix6d m = Matrix6d::Identity();
  m.topLeftCorner(3, 3) *= translational;
  m.bottomRightCorner(3, 3) *= rotational;
  return m;
}

}  // namespace

TEST(CartesianImpedanceLaw, ZeroErrorAtRestProducesZeroTaskTorque) {
  // The activation invariant. The controller seeds its equilibrium at the measured pose precisely
  // so this holds on the first update() — if it did not, switching controllers would kick the arm.
  const Pose here = makePose(0.4, 0.0, 0.5, 0.3);
  const Vector6d error = poseError(here, here);

  EXPECT_NEAR(error.norm(), 0.0, kTol);

  const Vector7d tau = taskTorque(wellConditionedJacobian(), error, Vector6d::Zero(),
                                  Vector7d::Zero(), diagonal(2000, 150), diagonal(89, 7),
                                  Matrix6d::Zero());
  EXPECT_NEAR(tau.norm(), 0.0, kTol);
}

TEST(CartesianImpedanceLaw, TorqueOpposesTheError) {
  // A spring pulls back. Displaced +x from the target, the resulting joint torque must produce an
  // end-effector force in -x. A sign flip here drives the arm away from every target it is given.
  const Vector6d error = poseError(makePose(0.41, 0.0, 0.5), makePose(0.40, 0.0, 0.5));
  EXPECT_GT(error(0), 0.0);

  const Jacobian j = wellConditionedJacobian();
  const Vector7d tau = taskTorque(j, error, Vector6d::Zero(), Vector7d::Zero(),
                                  diagonal(2000, 150), diagonal(89, 7), Matrix6d::Zero());

  // tau = J^T f, so recovering f from tau and checking its x component is the honest test.
  const Eigen::VectorXd force = forceFromTorque(j, tau);
  EXPECT_LT(force(0), 0.0);
}

TEST(CartesianImpedanceLaw, DampingOpposesVelocity) {
  // With zero error, moving must still be resisted, or the arm oscillates around every target.
  Vector7d dq = Vector7d::Zero();
  dq(0) = 1.0;

  const Jacobian j = wellConditionedJacobian();
  const Vector7d tau = taskTorque(j, Vector6d::Zero(), Vector6d::Zero(), dq, diagonal(2000, 150),
                                  diagonal(89, 7), Matrix6d::Zero());

  const Eigen::VectorXd force = forceFromTorque(j, tau);
  EXPECT_LT(force(0), 0.0);
}

TEST(CartesianImpedanceLaw, ClipCapsForceHoweverFarTheTargetIs) {
  // SERL's whole idea, and the property that makes a 2000 N/m spring safe to touch. A target 1 m
  // away must command exactly the same torque as one at the 1 cm clip — force stops growing.
  const ErrorClip clip;  // shipped defaults: 0.01 m, 0.05 rad
  const Pose current = makePose(0.40, 0.0, 0.5);

  const Vector6d at_clip = clipPoseError(poseError(current, makePose(0.39, 0.0, 0.5)), clip);
  const Vector6d far_away = clipPoseError(poseError(current, makePose(-0.60, 0.0, 0.5)), clip);

  EXPECT_NEAR((at_clip - far_away).norm(), 0.0, kTol);
  EXPECT_NEAR(at_clip(0), 0.01, kTol);

  // 2000 N/m * 0.01 m = 20 N, whatever the policy asked for.
  const Vector7d tau = taskTorque(wellConditionedJacobian(), far_away, Vector6d::Zero(),
                                  Vector7d::Zero(), diagonal(2000, 150), diagonal(89, 7),
                                  Matrix6d::Zero());
  const Eigen::VectorXd force = forceFromTorque(wellConditionedJacobian(), tau);
  EXPECT_NEAR(std::abs(force(0)), 20.0, 1e-9);
}

TEST(CartesianImpedanceLaw, ClipLeavesSmallErrorsUntouched) {
  // Accuracy is the other half of the claim: inside the clip the controller must be a plain
  // high-gain spring, not a saturated one.
  const ErrorClip clip;
  const Vector6d error = poseError(makePose(0.4020, 0.0, 0.5), makePose(0.4, 0.0, 0.5));

  EXPECT_NEAR(clipPoseError(error, clip)(0), error(0), kTol);
  EXPECT_NEAR(error(0), 0.002, kTol);
}

TEST(CartesianImpedanceLaw, DampedNullspaceLeakIsSmallButNotZero) {
  // What the shipped law actually does, pinned honestly. SERL's pseudo-inverse is damped
  // (lambda=0.2) to stay conditioned near singularities, and the price is that the projector is
  // only approximate: a couple of percent of the nullspace torque does reach the end effector.
  //
  // This is inherited behaviour, not a defect, and it is bounded — but it means the elbow term is
  // a small disturbance on the policy's tracking, not a free one. Raise nullspace_stiffness far
  // and this is what starts fighting the fingertips.
  const Jacobian j = wellConditionedJacobian();
  Vector7d q = Vector7d::Zero();
  q(6) = 0.3;

  const Vector7d tau = nullspaceTorque(j, q, Vector7d::Zero(), Vector7d::Zero(), 20.0, 100.0);

  ASSERT_GT(tau.norm(), 1.0) << "test is vacuous if the nullspace term is zero";
  const double leak_ratio = (j * tau).norm() / tau.norm();
  EXPECT_GT(leak_ratio, 1e-6) << "damped pinv should leak; an exact zero means lambda was dropped";
  EXPECT_LT(leak_ratio, 0.05);
}

TEST(CartesianImpedanceLaw, SaturationBoundsThePerCycleStep) {
  // libfranka rejects torque jumps. At 1 kHz this is the only thing between a bad interpolation
  // and a fault, so it must bound every joint, in both directions.
  Vector7d desired = Vector7d::Constant(50.0);
  desired(3) = -50.0;
  const Vector7d previous = Vector7d::Zero();

  const Vector7d out = saturateTorqueRate(desired, previous, 1.0);

  for (int i = 0; i < 7; ++i) {
    EXPECT_LE(std::abs(out(i) - previous(i)), 1.0 + kTol);
  }
  EXPECT_NEAR(out(0), 1.0, kTol);
  EXPECT_NEAR(out(3), -1.0, kTol);

  // The other half of the claim: a change already inside the bound passes through untouched, or
  // the controller would rate-limit its way through every ordinary command.
  EXPECT_NEAR((saturateTorqueRate(Vector7d::Constant(2.25), Vector7d::Constant(2.0), 1.0) -
               Vector7d::Constant(2.25))
                  .norm(),
              0.0, kTol);
}

TEST(CartesianImpedanceLaw, IntegralErrorCannotWindUp) {
  Vector6d integral = Vector6d::Zero();
  const Vector6d error = Vector6d::Constant(1.0);

  for (int i = 0; i < 1000; ++i) {
    integral = accumulateIntegralError(integral, error);
  }

  EXPECT_NEAR(integral.head(3).maxCoeff(), 0.1, kTol);
  EXPECT_NEAR(integral.tail(3).maxCoeff(), 0.3, kTol);
}

TEST(CartesianImpedanceLaw, ShiftedJacobianTurnsRotationIntoFingertipTranslation) {
  // Why the shift exists. Pure rotation about the EE moves a point 15 cm away; a Jacobian that
  // says otherwise makes the impedance blind to exactly the motion the fingertips feel.
  Jacobian j = Jacobian::Zero();
  j(5, 0) = 1.0;  // joint 0 produces angular velocity about base z at the EE

  const Eigen::Vector3d offset(0.0, 0.0, -0.1535);  // EE -> polyumi_tcp, roughly
  const Jacobian shifted = shiftJacobian(j, offset);

  // w x offset, with w = +z and offset along -z, is zero — collinear.
  // (Bound to a variable because the preprocessor reads the comma in block<3, 1> as an argument
  // separator.)
  const double linear_norm = shifted.block<3, 1>(0, 0).norm();
  EXPECT_NEAR(linear_norm, 0.0, kTol);

  // Offset perpendicular to the rotation axis does produce translation.
  const Jacobian perpendicular = shiftJacobian(j, Eigen::Vector3d(0.2, 0.0, 0.0));
  EXPECT_NEAR(perpendicular(1, 0), 0.2, kTol);  // z_hat x 0.2 x_hat = 0.2 y_hat
  EXPECT_NEAR(perpendicular(5, 0), 1.0, kTol);  // angular rows untouched
}

TEST(CartesianImpedanceLaw, RotationErrorTakesTheShortWayRound) {
  // A target 10 degrees back must read as -10, not +350; otherwise the arm unwinds the long way
  // through its joint limits.
  const Pose current = makePose(0.4, 0.0, 0.5, 350.0 * M_PI / 180.0);
  const Pose desired = makePose(0.4, 0.0, 0.5, 0.0);

  const Vector6d error = poseError(current, desired);

  // |vec(q_err)| = sin(angle/2); 10 degrees -> sin(5 deg) ~ 0.0872.
  EXPECT_NEAR(error.tail(3).norm(), std::sin(5.0 * M_PI / 180.0), 1e-9);
}

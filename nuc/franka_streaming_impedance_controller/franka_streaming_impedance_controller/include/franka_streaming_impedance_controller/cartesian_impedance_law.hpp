// Copyright (c) 2026 the franka_streaming_impedance_controller authors. MIT.
//
// The Cartesian impedance control law, ported from
// https://github.com/rail-berkeley/serl_franka_controllers (MIT) — itself derived from
// franka_ros's cartesian_impedance_example_controller.
//
// This is the same law polymetis runs for UMI (torchcontrol OperationalSpacePD:
// `tau = J^T (Kp*e + Kd*(-J*dq)) + coriolis`), with the error sign convention flipped and two
// additions kept from SERL: a per-axis error clip, and a nullspace term, without which a 7-DOF
// elbow drifts.
//
// Free functions over plain Eigen arguments rather than methods on the controller: the law is the
// part worth unit-testing, and it must not need a robot to run.

#pragma once

#include <franka_streaming_impedance_controller/pose_trajectory_interpolator.hpp>

#include <Eigen/Dense>

namespace franka_streaming_impedance {

using Vector7d = Eigen::Matrix<double, 7, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Jacobian = Eigen::Matrix<double, 6, 7>;

/// Per-axis bounds on the pose error, in metres and radians. Defaults are SERL's.
struct ErrorClip {
  Eigen::Vector3d translation_min{Eigen::Vector3d::Constant(-0.01)};
  Eigen::Vector3d translation_max{Eigen::Vector3d::Constant(0.01)};
  Eigen::Vector3d rotation_min{Eigen::Vector3d::Constant(-0.05)};
  Eigen::Vector3d rotation_max{Eigen::Vector3d::Constant(0.05)};
};

/**
 * Pose error as a 6-vector in the base frame, ordered (translation, rotation).
 *
 * Sign convention is SERL's: current MINUS desired, so the task torque below negates it. Rotation
 * error is the vector part of the difference quaternion, hemisphere-corrected so a 350-degree error
 * reads as -10 rather than +350, then rotated into the base frame to match the Jacobian.
 */
Vector6d poseError(const Pose& current, const Pose& desired);

/**
 * Clamp each axis of the error independently, which is what bounds commanded force.
 *
 * Force is `stiffness * error`, so a bounded error is a bounded force however far the equilibrium
 * point has run away. Why the shipped numbers are what they are: the example controllers.yaml.
 */
Vector6d clipPoseError(const Vector6d& error, const ErrorClip& clip);

/**
 * Accumulate the integral error, clamped to SERL's 0.1 translation / 0.3 rotation so it cannot
 * wind up. It accumulates even at the default zero Ki, so the clamp still matters if Ki is turned
 * on mid-run.
 */
Vector6d accumulateIntegralError(const Vector6d& integral, const Vector6d& error);

/// `tau = J^T (-K e - D (J dq) - Ki e_i)`, the impedance itself.
Vector7d taskTorque(const Jacobian& jacobian,
                    const Vector6d& error,
                    const Vector6d& integral_error,
                    const Vector7d& dq,
                    const Matrix6d& stiffness,
                    const Matrix6d& damping,
                    const Matrix6d& ki);

/**
 * Joint-space pull toward `q_nullspace`, projected into the Jacobian's null space, so it resolves
 * the 7th DOF without disturbing the end effector. `joint1_stiffness` multiplies joint 1 alone,
 * pinning the base rotation while the elbow stays free.
 *
 * The projector uses a DAMPED pseudo-inverse, so the isolation is approximate — see
 * DampedNullspaceLeakIsSmallButNotZero in the tests for how much leaks through.
 */
Vector7d nullspaceTorque(const Jacobian& jacobian,
                         const Vector7d& q,
                         const Vector7d& dq,
                         const Vector7d& q_nullspace,
                         double stiffness,
                         double joint1_stiffness);

/**
 * Bound the per-cycle change in commanded torque against what was last commanded.
 *
 * libfranka rejects torque jumps outright; at 1 kHz, 1.0 Nm/cycle is SERL's limit. `tau_previous`
 * must be the last DESIRED torque, never the measured one — chaining against the measurement would
 * let the bound drift with load. The controller passes its own last command.
 */
Vector7d saturateTorqueRate(const Vector7d& tau_desired,
                            const Vector7d& tau_previous,
                            double max_delta);

/**
 * Move a base-frame ("zero") Jacobian from one rigidly-attached point to another.
 *
 * `offset` is the base-frame vector from the current point to the new one; `v_new = v_old + w x
 * offset`, and the angular rows are unchanged. Only the translation matters, because a zero
 * Jacobian already expresses both velocities in the base frame.
 *
 * Needed because franka reports its Jacobian at `<arm_id>_hand_tcp` while the contact happens at
 * `tcp_frame`, which for a custom tool is some distance away — a spring anchored at the wrong
 * point turns an orientation error into a large translation at the actual contact point.
 */
Jacobian shiftJacobian(const Jacobian& jacobian, const Eigen::Vector3d& offset);

/// Damped pseudo-inverse (SVD, lambda=0.2), as SERL's pseudo_inversion.h computes it.
Eigen::MatrixXd dampedPseudoInverse(const Eigen::MatrixXd& m, double lambda = 0.2);

}  // namespace franka_streaming_impedance

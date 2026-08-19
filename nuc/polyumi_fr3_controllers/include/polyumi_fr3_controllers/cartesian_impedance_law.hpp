// Copyright (c) 2026 PolyUMI. MIT.
//
// The Cartesian impedance control law, ported from
// https://github.com/rail-berkeley/serl_franka_controllers (MIT) — itself derived from
// franka_ros's cartesian_impedance_example_controller.
//
// This is the same law polymetis runs for UMI (torchcontrol OperationalSpacePD:
// `tau = J^T (Kp*e + Kd*(-J*dq)) + coriolis`), with the error sign convention flipped and two
// additions kept from SERL: a per-axis error clip, which caps commanded force while leaving the
// stiffness high enough to track accurately, and a nullspace term, which polymetis lacks entirely
// and without which a 7-DOF elbow drifts.
//
// Deliberately free functions over plain Eigen arguments rather than methods on the controller:
// the law is the part worth unit-testing, and it must not need a robot to run.

#pragma once

#include <polyumi_fr3_controllers/pose_trajectory_interpolator.hpp>

#include <Eigen/Dense>

namespace polyumi_fr3_controllers {

using Vector7d = Eigen::Matrix<double, 7, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Jacobian = Eigen::Matrix<double, 6, 7>;

/// Per-axis bounds on the pose error, in metres and radians. See clipPoseError.
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
 * Clamp each axis of the error independently.
 *
 * SERL's contribution and the reason this controller can be both stiff and safe: force is
 * `stiffness * error`, so bounding the error bounds the force regardless of how far the
 * equilibrium point has run away. At the shipped 2000 N/m and 0.01 m that is ~20 N. Without it a
 * policy commanding an unreachable pose leans on the environment with everything the arm has.
 */
Vector6d clipPoseError(const Vector6d& error, const ErrorClip& clip);

/**
 * Accumulate the integral error, clamped so it cannot wind up.
 *
 * Bounds are SERL's: 0.1 on translation, 0.3 on rotation. With the default zero Ki this term
 * contributes nothing, but it still accumulates, so the clamp is what keeps it bounded if Ki is
 * ever turned on mid-run.
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
 * Joint-space pull toward `q_nullspace`, projected into the Jacobian's null space.
 *
 * Being in the null space, it cannot disturb the end-effector — it only resolves the 7th DOF.
 * `joint1_stiffness` is a multiplier on joint 1 alone (SERL runs it ~500x the rest), which pins the
 * base rotation while leaving the elbow free.
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
 * libfranka rejects torque jumps outright; at 1 kHz, 1.0 Nm/cycle is SERL's limit. Note this is
 * measured against `tau_J_d` (the last DESIRED torque), not the measured one — chaining against
 * the measurement would let the bound drift with load.
 */
Vector7d saturateTorqueRate(const Vector7d& tau_desired,
                            const Vector7d& tau_previous,
                            double max_delta);

/**
 * Move a base-frame ("zero") Jacobian from one rigidly-attached point to another.
 *
 * `offset` is the base-frame vector from the current point to the new one. Only the translation
 * matters: a zero Jacobian already expresses both linear and angular velocity in the base frame,
 * so the new point's rotation offset changes nothing, and `v_new = v_old + w x offset`.
 *
 * We need this because franka reports `O_T_EE` and its Jacobian at `fr3_hand_tcp`, while the
 * policy — and the physical contact — happen at `polyumi_tcp`, ~15 cm away. Anchoring the spring
 * at the wrong point makes an orientation error show up as a large fingertip translation.
 */
Jacobian shiftJacobian(const Jacobian& jacobian, const Eigen::Vector3d& offset);

/// Damped pseudo-inverse (SVD, lambda=0.2), as SERL's pseudo_inversion.h computes it.
Eigen::MatrixXd dampedPseudoInverse(const Eigen::MatrixXd& m, double lambda = 0.2);

}  // namespace polyumi_fr3_controllers

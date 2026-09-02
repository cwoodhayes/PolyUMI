// Copyright (c) 2026 the franka_streaming_impedance_controller authors. MIT.
// Ported from serl_franka_controllers (MIT), rail-berkeley.

#include <franka_streaming_impedance_controller/cartesian_impedance_law.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>

namespace franka_streaming_impedance {

Vector6d poseError(const Pose& current, const Pose& desired) {
  Vector6d error;
  error.head(3) = current.position - desired.position;

  // Take the shorter arc: without this a rotation error just under a full turn reads as nearly a
  // full turn, and the arm spins the long way round.
  Eigen::Quaterniond orientation = current.orientation;
  if (desired.orientation.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() = -orientation.coeffs();
  }

  const Eigen::Quaterniond difference = orientation.inverse() * desired.orientation;
  error.tail(3) = difference.vec();
  // difference is in the end-effector frame; the Jacobian is in the base frame.
  error.tail(3) = -(orientation.toRotationMatrix() * error.tail(3));
  return error;
}

Vector6d clipPoseError(const Vector6d& error, const ErrorClip& clip) {
  Vector6d clipped = error;
  for (int i = 0; i < 3; ++i) {
    clipped(i) = std::clamp(error(i), clip.translation_min(i), clip.translation_max(i));
    clipped(i + 3) = std::clamp(error(i + 3), clip.rotation_min(i), clip.rotation_max(i));
  }
  return clipped;
}

Vector6d accumulateIntegralError(const Vector6d& integral, const Vector6d& error) {
  Vector6d out = integral + error;
  out.head(3) = out.head(3).cwiseMax(-0.1).cwiseMin(0.1);
  out.tail(3) = out.tail(3).cwiseMax(-0.3).cwiseMin(0.3);
  return out;
}

Vector7d taskTorque(const Jacobian& jacobian,
                    const Vector6d& error,
                    const Vector6d& integral_error,
                    const Vector7d& dq,
                    const Matrix6d& stiffness,
                    const Matrix6d& damping,
                    const Matrix6d& ki) {
  return jacobian.transpose() *
         (-stiffness * error - damping * (jacobian * dq) - ki * integral_error);
}

Vector7d nullspaceTorque(const Jacobian& jacobian,
                         const Vector7d& q,
                         const Vector7d& dq,
                         const Vector7d& q_nullspace,
                         double stiffness,
                         double joint1_stiffness) {
  Vector7d qe = q_nullspace - q;
  Vector7d dqe = dq;
  // Joint 1 gets its own tier, so the base rotation is held firmly while the elbow stays free.
  qe(0) *= joint1_stiffness;
  dqe(0) *= 2.0 * std::sqrt(joint1_stiffness);

  const Eigen::MatrixXd jacobian_transpose_pinv = dampedPseudoInverse(jacobian.transpose());
  const Eigen::Matrix<double, 7, 7> projector =
      Eigen::Matrix<double, 7, 7>::Identity() - jacobian.transpose() * jacobian_transpose_pinv;

  return projector * (stiffness * qe - (2.0 * std::sqrt(stiffness)) * dqe);
}

Vector7d saturateTorqueRate(const Vector7d& tau_desired,
                            const Vector7d& tau_previous,
                            double max_delta) {
  Vector7d saturated;
  for (int i = 0; i < 7; ++i) {
    const double difference = tau_desired(i) - tau_previous(i);
    saturated(i) = tau_previous(i) + std::clamp(difference, -max_delta, max_delta);
  }
  return saturated;
}

Jacobian shiftJacobian(const Jacobian& jacobian, const Eigen::Vector3d& offset) {
  Jacobian shifted = jacobian;
  // v_new = v_old + w x offset; angular rows are unchanged.
  for (int col = 0; col < 7; ++col) {
    shifted.block<3, 1>(0, col) =
        jacobian.block<3, 1>(0, col) + jacobian.block<3, 1>(3, col).cross(offset);
  }
  return shifted;
}

Eigen::MatrixXd dampedPseudoInverse(const Eigen::MatrixXd& m, double lambda) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto& singular = svd.singularValues();

  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(m.rows(), m.cols());
  for (int i = 0; i < singular.size(); ++i) {
    s(i, i) = singular(i) / (singular(i) * singular(i) + lambda * lambda);
  }

  return svd.matrixV() * s.transpose() * svd.matrixU().transpose();
}

}  // namespace franka_streaming_impedance

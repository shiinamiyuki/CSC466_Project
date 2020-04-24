#include "end_effectors_objective_and_gradient.h"
#include "copy_skeleton_at.h"
#include "kinematics_jacobian.h"
#include "transformed_tips.h"
#include <iostream>

void end_effectors_objective_and_gradient(
    const Skeleton &skeleton, const Eigen::VectorXi &b,
    const Eigen::VectorXd &xb0,
    std::function<double(const Eigen::VectorXd &)> &f,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &grad_f,
    std::function<void(Eigen::VectorXd &)> &proj_z) {
  /////////////////////////////////////////////////////////////////////////////
  // Replace with your code
  f = [=, &skeleton, &b](const Eigen::VectorXd &A) -> double {
    auto cp = copy_skeleton_at(skeleton, A);
    auto x = transformed_tips(cp, b);
    double E = 0.5 * (x - xb0).squaredNorm();
    return E;
  };
  grad_f = [=, &skeleton, &b](const Eigen::VectorXd &A) -> Eigen::VectorXd {
    Eigen::MatrixXd J;
    auto cp = copy_skeleton_at(skeleton, A);
    kinematics_jacobian(cp, b, J);
    auto x = transformed_tips(cp, b);
    auto dEdx = (x - xb0);
    return J.transpose() * dEdx;
  };
  proj_z = [=, &skeleton, &b](Eigen::VectorXd &A) {
    assert(skeleton.size() * 3 == A.size());
    for (size_t i = 0; i < skeleton.size(); i++) {
      for (int k = 0; k < 3; k++) {
        A(3 * i + k) = std::min(std::max(skeleton[i].xzx_min[k], A(3 * i + k)),
                                skeleton[i].xzx_max[k]);
      }
    }
  };
  /////////////////////////////////////////////////////////////////////////////
}

void end_effectors_objective_and_gradient_and_jacobian(
    const Skeleton &skeleton, const Eigen::VectorXi &b,
    const Eigen::VectorXd &xb0,
    std::function<double(const Eigen::VectorXd &)> &f,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &grad_f,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &dEdx,
    std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> &jacobian_f,
    std::function<void(Eigen::VectorXd &)> &proj_z) {
  end_effectors_objective_and_gradient(skeleton, b, xb0, f, grad_f, proj_z);
  dEdx = [=, &skeleton, &b](const Eigen::VectorXd &A) -> Eigen::VectorXd {
    auto cp = copy_skeleton_at(skeleton, A);
    auto x = transformed_tips(cp, b);
    return (x - xb0);
  };
  jacobian_f = [=, &skeleton, &b](const Eigen::VectorXd &A) -> Eigen::MatrixXd {
    Eigen::MatrixXd J;
    auto cp = copy_skeleton_at(skeleton, A);
    kinematics_jacobian(cp, b, J);
    return J;
  };
}
#include "kinematics_jacobian.h"
#include "copy_skeleton_at.h"
#include "transformed_tips.h"
#include <iostream>

void kinematics_jacobian(const Skeleton &skeleton, const Eigen::VectorXi &b,
                         Eigen::MatrixXd &J) {
  /////////////////////////////////////////////////////////////////////////////
  // Replace with your code
  double h = 1e-7;
  J = Eigen::MatrixXd::Zero(b.size() * 3, skeleton.size() * 3);
  auto cp = skeleton;
  for (int j = 0; j < skeleton.size(); j++) {
    for (int k = 0; k < 3; k++) {
      auto tmp = cp[j].xzx;
      auto tmp_xzx = cp[j].xzx(k);
      cp[j].xzx(k) += h;
      cp[j].xzx(k) = std::fmin(std::fmax(cp[j].xzx(k), cp[j].xzx_min(k)),  cp[j].xzx_max(k));
      auto x_1 = transformed_tips(cp, b);
      cp[j].xzx(k) = tmp_xzx;
      cp[j].xzx(k) -= h;
      cp[j].xzx(k) = std::fmin(std::fmax(cp[j].xzx(k), cp[j].xzx_min(k)),  cp[j].xzx_max(k));
      auto x = transformed_tips(cp, b);
      auto diff = (x_1 - x) / h;
      J.col(3 * j + k) = diff;
      cp[j].xzx = tmp;
    }
  }
  //  for (int i = 0; i < J.rows(); i++) {
  //    for (int j = 0; j < J.cols(); j++) {
  //      printf("%3.5lf ", J.row(i)(j));
  //    }
  //    printf("\n");
  //  }
  /////////////////////////////////////////////////////////////////////////////
}

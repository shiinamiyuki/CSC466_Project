#include "projected_gradient_descent.h"
#include "line_search.h"
#include <Eigen/Dense>

static double backtracking(
    double c1, double k, const Eigen::VectorXd &x, const Eigen::VectorXd &p,
    const std::function<double(const Eigen::VectorXd &)> &f,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &grad_f,
    const std::function<void(Eigen::VectorXd &)> &proj_z) {
  double alpha = 1;
  while (alpha > 1e-7) {
    Eigen::VectorXd z = x + alpha * p;
    proj_z(z);
    if (f(z) < f(x)) {
      break;
    }
    alpha *= k;
  }
  return alpha;
}

void projected_gradient_descent(
    const std::function<double(const Eigen::VectorXd &)> &f,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &grad_f,
    const std::function<void(Eigen::VectorXd &)> &proj_z, const int max_iters,
    Eigen::VectorXd &z) {
  /////////////////////////////////////////////////////////////////////////////
  const double max_step = 1;
  //  Eigen::MatrixXd B = Eigen::MatrixXd::Identity(z.rows(), z.rows());
  Eigen::MatrixXd BI = Eigen::MatrixXd::Identity(z.rows(), z.rows());
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(z.rows(), z.rows());
  for (int i = 0; i < max_iters; i++) {
    Eigen::VectorXd p = -BI * grad_f(z);
    double alpha = backtracking(0.001, 0.5, z, p, f, grad_f, proj_z);
    if (alpha < 1e-7) {
      //      printf("iter:%d\n",i+1);
      break;
    }
    Eigen::VectorXd next_z = z + alpha * p;
    proj_z(next_z);
    Eigen::VectorXd y = grad_f(next_z) - grad_f(z);
    Eigen::VectorXd s = next_z - z;

    BI = (I - (s * y.transpose()) / (y.transpose() * s)) * BI *
             (I - (y * s.transpose()) / (y.transpose() * s)) +
         (s * s.transpose()) / (y.transpose() * s);
    z = next_z;
    //    B = B + (y * y.transpose()) / (y.transpose() * s) -
    //        (B * s * s.transpose() * B.transpose()) / (s.transpose() * B * s);
    //                auto grad = grad_f(z);
    //                auto step = line_search(f, proj_z, z, grad, max_step);
    //                if(step == 0.0f)break;
    //
    //                z -= step * grad;
    //                proj_z(z);

    //          proj_z(z);
  }
  //  proj_z(z);

  /////////////////////////////////////////////////////////////////////////////
}

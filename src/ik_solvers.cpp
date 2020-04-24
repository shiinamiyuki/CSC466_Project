#include <Eigen/Dense>
#include <IKSolver.h>
#include <end_effectors_objective_and_gradient.h>
#include <igl/pinv.h>
#include <line_search.h>

static double backtracking(
    double c1, double k, const Eigen::VectorXd &x, const Eigen::VectorXd &p,
    const std::function<double(const Eigen::VectorXd &)> &f,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &grad_f,
    const std::function<void(Eigen::VectorXd &)> &proj_z) {
  double alpha = 1;
  Eigen::VectorXd g0 = grad_f(x);
  while (alpha > 0) {
    Eigen::VectorXd z = x + alpha * p;
    proj_z(z);
    // This is enough to guarantee sufficient decrease
    //    printf("f(z);%lf %lf\n",f(z), f(x) + c1 * alpha * g0.dot(p));
    if (f(z) < f(x) + c1 * alpha * g0.dot(p)) {
      break;
    }
    alpha *= k;
  }
  return alpha;
}

/* The projected gradient solver
 * using the same line search strategy as in the original assignment
 * */
class GradientDescentSolver : public IKSolver {
  std::function<double(const Eigen::VectorXd &)> f;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f;
  std::function<void(Eigen::VectorXd &)> proj_z;
  Eigen::VectorXd z;
  bool reached = true;
  size_t nIterations = 0;
  Tolerance tol;

public:
  void initialize(const Skeleton &skeleton, const Eigen::VectorXi &b,
                  const Eigen::VectorXd &xb0, const Eigen::VectorXd &_z,
                  Tolerance tolerance) override {
    printf("Intialize gradient descent solver\n");
    tol = tolerance;
    this->z = _z;
    reached = false;
    nIterations = 0;
    end_effectors_objective_and_gradient(skeleton, b, xb0, f, grad_f, proj_z);
  }
  void do_iteration() override {
    if (reached) {
      return;
    }
    nIterations++;
    const double max_step = 180 / M_PI; // 1e5;//180 / M_PI;
    auto grad = grad_f(z);
    auto step = line_search(f, proj_z, z, grad, max_step);
    if (step == 0.0f || grad.norm() < tol.grad_tolerance ||
        f(z) < tol.f_tolerance) {
      reached = true;
      printf("f:%lf g:%lf\n", tol.f_tolerance, tol.grad_tolerance);
      printf("f:%lf g:%lf\n", f(z), grad.norm());
      printf("took %zd iterations\n", nIterations);
      return;
    }
    z -= step * grad;
    proj_z(z);
  }
  Eigen::VectorXd &get_z() override { return z; }
  bool has_reached_target() const override { return reached; }
  double get_f() const override { return f(z); }
};

std::shared_ptr<IKSolver> create_gradient_descent_solver() {
  return std::make_shared<GradientDescentSolver>();
}

/*
 * BFGS projected gradient solver
 *
 * BI keeps track of B_k^{-1}
 *
 * */
class BFGSSolver : public IKSolver {
  std::function<double(const Eigen::VectorXd &)> f;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f;
  std::function<void(Eigen::VectorXd &)> proj_z;
  Eigen::VectorXd z;
  bool reached = true;
  size_t nIterations = 0;
  Eigen::MatrixXd BI, B;
  Eigen::MatrixXd I;
  Tolerance tol;

public:
  void initialize(const Skeleton &skeleton, const Eigen::VectorXi &b,
                  const Eigen::VectorXd &xb0, const Eigen::VectorXd &_z,
                  Tolerance tolerance) override {
    printf("Intialize BFGS solver\n");
    this->z = _z;
    reached = false;
    nIterations = 0;
    end_effectors_objective_and_gradient(skeleton, b, xb0, f, grad_f, proj_z);
    BI = B = Eigen::MatrixXd::Identity(z.rows(), z.rows());
    I = Eigen::MatrixXd::Identity(z.rows(), z.rows());
    tol = tolerance;
  }
  void do_iteration() override {
    if (reached) {
      return;
    }
    nIterations++;
    Eigen::VectorXd grad = grad_f(z);
    Eigen::VectorXd p = -BI * grad;
    if (p.dot(grad) > 0) {
      fprintf(stderr, "B not positive definite\n");
    }
    double alpha = backtracking(0.001, 0.5, z, p, f, grad_f, proj_z);
    //    printf("%lf %lf %lf\n", alpha, grad.norm(), grad.dot(p));
    if (alpha == 0 || grad.norm() < tol.grad_tolerance ||
        f(z) < tol.f_tolerance) {
      reached = true;
      printf("took %zd iterations\n", nIterations);
      return;
    }
    Eigen::VectorXd next_z = z + alpha * p;
    proj_z(next_z);
    Eigen::VectorXd y = grad_f(next_z) - grad_f(z);
    Eigen::VectorXd s = next_z - z;
    double sBs = s.transpose() * B * s;
    double theta;
    if (s.dot(y) >= 0.2 * sBs) {
      theta = 1;
    } else {
      theta = (0.8 * sBs) / (sBs - s.dot(y));
    }
    Eigen::VectorXd r = theta * y + (1 - theta) * B * s;
    if (s.dot(r) <= 0) {
      fprintf(stderr, "curvature condition not satisfied\n");
    }
    BI = (I - (s * r.transpose()) / (r.transpose() * s)) * BI *
             (I - (r * s.transpose()) / (r.transpose() * s)) +
         (s * s.transpose()) / (r.transpose() * s);
    B = B + (r * r.transpose()) / (r.transpose() * s) -
        (B * s * s.transpose() * B.transpose()) / (s.transpose() * B * s);
    z = next_z;
  }
  Eigen::VectorXd &get_z() override { return z; }
  bool has_reached_target() const override { return reached; }
  double get_f() const override { return f(z); }
};
std::shared_ptr<IKSolver> create_BFGS_solver() {
  return std::make_shared<BFGSSolver>();
}

/* Newton solver by approximating Hessian using brute force
 *
 * Almost always Hessian is singular
 * */
class NewtonSolver : public IKSolver {
  std::function<double(const Eigen::VectorXd &)> f;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f;
  std::function<void(Eigen::VectorXd &)> proj_z;
  Eigen::VectorXd z;
  bool reached = true;
  size_t nIterations = 0;
  Tolerance tol;

public:
  Eigen::MatrixXd hessian(const Eigen::VectorXd &x) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(x.rows(), x.rows());
    double h = 5e-3;
    Eigen::VectorXd g0 = grad_f(x);
    for (size_t i = 0; i < x.rows(); i++) {
      Eigen::VectorXd x1 = x;
      x1(i) += h;
      auto g1 = grad_f(x1);
      //      printf("%.10e\n", ((g1 - g0).norm()));
      H.col(i) = (g1 - g0) / h;
    }
    return H;
  }

  void initialize(const Skeleton &skeleton, const Eigen::VectorXi &b,
                  const Eigen::VectorXd &xb0, const Eigen::VectorXd &_z,
                  Tolerance tolerance) override {
    this->z = _z;
    reached = false;
    nIterations = 0;
    end_effectors_objective_and_gradient(skeleton, b, xb0, f, grad_f, proj_z);
  }
  void do_iteration() override {
    if (reached) {
      return;
    }
    Eigen::MatrixXd H = hessian(z);
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(H);
    if (!lu_decomp.isInvertible()) {
      reached = true;
      printf("Hessian not invertible\n");
      return;
    }
    // grad + Hp = 0
    // Hp = -grad
    Eigen::VectorXd grad = grad_f(z);
    Eigen::VectorXd p = lu_decomp.solve(-grad);
    double alpha = backtracking(0.001, 0.5, z, p, f, grad_f, proj_z);
    if (alpha == 0 || grad.norm() < tol.grad_tolerance ||
        f(z) < tol.f_tolerance) {
      reached = true;
      printf("took %zd iterations\n", nIterations);
      return;
    }
    z = z + alpha * p;
    proj_z(z);
    nIterations++;
  }
  Eigen::VectorXd &get_z() override { return z; }
  bool has_reached_target() const override { return reached; }
  double get_f() const override { return f(z); }
};
std::shared_ptr<IKSolver> create_Newton_solver() {
  return std::make_shared<NewtonSolver>();
}

class GaussNewtonSolver : public IKSolver {
  std::function<double(const Eigen::VectorXd &)> f;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_f;
  std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> jacobian_f;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> dEdx;
  std::function<void(Eigen::VectorXd &)> proj_z;
  Eigen::VectorXd z;
  bool reached = true;
  size_t nIterations = 0;
  Tolerance tol;

public:
  void initialize(const Skeleton &skeleton, const Eigen::VectorXi &b,
                  const Eigen::VectorXd &xb0, const Eigen::VectorXd &_z,
                  Tolerance tolerance) override {
    printf("Initialize Gauss Newton solver\n");
    tol = tolerance;
    this->z = _z;
    reached = false;
    nIterations = 0;
    end_effectors_objective_and_gradient_and_jacobian(
        skeleton, b, xb0, f, grad_f, dEdx, jacobian_f, proj_z);
  }
  void do_iteration() override {
    if (reached) {
      return;
    }
    Eigen::MatrixXd J = jacobian_f(z);

    Eigen::MatrixXd pinv_J;
    igl::pinv(J, pinv_J);
    Eigen::VectorXd grad = grad_f(z);
    Eigen::VectorXd p = -pinv_J * dEdx(z);
    //    printf("|grad| %lf\n", grad.norm());
    //    printf("p.dot(grad) %.5e\n", p.dot(grad));
    double alpha = backtracking(0.001, 0.5, z, p, f, grad_f, proj_z);
    //    printf("alpha %lf\n", alpha);
    if (alpha == 0 || grad.norm() < tol.grad_tolerance ||
        f(z) < tol.f_tolerance) {
      reached = true;
      printf("took %zd iterations\n", nIterations);
      return;
    }
    z = z + alpha * p;
    proj_z(z);
    nIterations++;
  }
  Eigen::VectorXd &get_z() override { return z; }
  bool has_reached_target() const override { return reached; }
  double get_f() const override { return f(z); }
};
std::shared_ptr<IKSolver> create_Gauss_Newton_solver() {
  return std::make_shared<GaussNewtonSolver>();
}
#ifndef KINEMATICS_IKSOLVER_H
#define KINEMATICS_IKSOLVER_H

#include "Skeleton.h"
#include <Eigen/Core>
#include <chrono>
#include <functional>
#include <memory>

class Timer {
  decltype(std::chrono::system_clock::now()) start = std::chrono::system_clock::now();
public:
  [[nodiscard]] double elapsed_seconds()const{
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = now - start;
    return elapsed.count();
  }
};

struct Tolerance {
  double grad_tolerance = 5e-3;
  double f_tolerance = 0.01;
};

class IKSolver {
public:
  // b is indices of end effectors and xb0 is target end effector pos
  virtual void initialize(const Skeleton &skeleton, const Eigen::VectorXi &b,
                          const Eigen::VectorXd &xb0,
                          const Eigen::VectorXd &z, Tolerance tolerance) = 0;

  // to one iteration
  virtual void do_iteration() = 0;
  virtual Eigen::VectorXd &get_z() = 0;
  // The end effectors has reached target or it is not reachable but a local
  // minimum is found
  virtual bool has_reached_target() const = 0;

  virtual double get_f() const = 0;
};

std::shared_ptr<IKSolver> create_gradient_descent_solver();
std::shared_ptr<IKSolver> create_BFGS_solver();
std::shared_ptr<IKSolver> create_Newton_solver();
std::shared_ptr<IKSolver> create_Gauss_Newton_solver();

#endif // KINEMATICS_IKSOLVER_H

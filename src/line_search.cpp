#include "line_search.h"
#include <iostream>

double line_search(
        const std::function<double(const Eigen::VectorXd &)> &f,
        const std::function<void(Eigen::VectorXd &)> &proj_z,
        const Eigen::VectorXd &z,
        const Eigen::VectorXd &dz,
        const double max_step) {
    /////////////////////////////////////////////////////////////////////////////
    auto step = max_step;
    auto Ea = f(z);
    while (step > 0) {
        Eigen::VectorXd a = z - step * dz;
        proj_z(a);
        if (f(a) < Ea) {
            break;
        }
        step /= 2.0;
    }
    // printf("step: %f Ea: %f\n",step, Ea);
    return step;
    /////////////////////////////////////////////////////////////////////////////
}

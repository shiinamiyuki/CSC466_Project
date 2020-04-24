#include "transformed_tips.h"
#include "forward_kinematics.h"

Eigen::VectorXd transformed_tips(
        const Skeleton &skeleton,
        const Eigen::VectorXi &b) {
    /////////////////////////////////////////////////////////////////////////////
    // Replace with your code
    std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d> > T;
    forward_kinematics(skeleton, T);
    auto result = Eigen::VectorXd(3 * b.size());
    for (int i = 0; i < b.size(); i++) {
        auto idx = b[i];
        Eigen::Vector3d tip = T[idx] * skeleton[idx].rest_T * Eigen::Vector3d(skeleton[idx].length, 0, 0);
        result(3 * i + 0) = tip[0];
        result(3 * i + 1) = tip[1];
        result(3 * i + 2) = tip[2];
        //result << tip[0],tip[1], tip[2];
    }
    return result;
    /////////////////////////////////////////////////////////////////////////////
}

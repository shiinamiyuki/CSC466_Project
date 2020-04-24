#include "forward_kinematics.h"
#include "euler_angles_to_transform.h"
#include <functional> // std::function

void forward_kinematics(
        const Skeleton &skeleton,
        std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d> > &T) {
    /////////////////////////////////////////////////////////////////////////////
    // Replace with your code
    std::vector<int> visited;
    visited.resize(skeleton.size(), 0);

    T.resize(skeleton.size(), Eigen::Affine3d::Identity());
    std::function<void(int)> trace;
    trace = [&](int i) {
        if (visited[i])return;
        auto &bone = skeleton[i];
        auto M = bone.rest_T * euler_angles_to_transform(bone.xzx) * bone.rest_T.inverse();
        if (-1 != bone.parent_index) {
            trace(bone.parent_index);
            T[i] = T[bone.parent_index] * M;
        } else {
            T[i] = M;
        }
        visited[i] = true;
    };
    for (int i = 0; i < skeleton.size(); i++) {
        trace(i);
    }
    /////////////////////////////////////////////////////////////////////////////
}

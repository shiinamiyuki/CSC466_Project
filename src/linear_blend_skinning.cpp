#include "linear_blend_skinning.h"

void linear_blend_skinning(
        const Eigen::MatrixXd &V,
        const Skeleton &skeleton,
        const std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d> > &T,
        const Eigen::MatrixXd &W,
        Eigen::MatrixXd &U) {
    /////////////////////////////////////////////////////////////////////////////
    // Replace with your code
   // U = V;return;
    U.resize(V.rows(), V.cols());
    for (int i = 0; i < V.rows(); i++) {
        Eigen::Vector3d v(0, 0, 0);
        for (int m = 0; m < skeleton.size(); m++) {
            if(skeleton[m].weight_index!=-1)
                v += W(i, skeleton[m].weight_index) * (T[m] * Eigen::Vector3d(V.row(i)));
        }
        U.row(i) = v;
    }

    /////////////////////////////////////////////////////////////////////////////
}

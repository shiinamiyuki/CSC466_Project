#include "catmull_rom_interpolation.h"
#include <Eigen/Dense>

Eigen::Vector3d catmull_rom_interpolation(
        const std::vector<std::pair<double, Eigen::Vector3d> > &keyframes,
        double t) {
    /////////////////////////////////////////////////////////////////////////////
    if (keyframes.empty()) {
        return Eigen::Vector3d(0, 0, 0);
    }
    if(keyframes.size() == 1){
        return keyframes[0].second;
    }
    int index = 0;
    for (const auto &frame : keyframes) {
        auto time = frame.first;
        if (time > t) {
            --index;
            break;
        }
        index++;
    }
    auto s = (t - keyframes[index].first) / (keyframes[index + 1].first - keyframes[index].first);
    if(index >= keyframes.size() - 1){
        return keyframes.back().second;
    }

    double c = 0.5;
    auto compute_tangent = [&](int k) -> Eigen::Vector3d {
        if (k <= 0 || k >= keyframes.size() - 1) {
            return Eigen::Vector3d(0, 0, 0);
        }
        return (1 - c)
               * (keyframes[k + 1].second - keyframes[k - 1].second)
               / (keyframes[k + 1].first - keyframes[k - 1].first);
    };
    Eigen::Matrix4d m;
    m << 0, 1, 0, 3,
            0, 1, 0, 2,
            0, 1, 1, 1,
            1, 1, 0, 0;
    m = m.inverse();
    Eigen::MatrixX4d v;
    v.resize(3, 4);

    v << keyframes[index].second,
            keyframes[index + 1].second,
            compute_tangent(index),
            compute_tangent(index + 1);
    auto poly = v * m;

    auto S = Eigen::Vector4d(s * s * s, s * s, s, 1.0);
    return poly * S;
    /////////////////////////////////////////////////////////////////////////////
}

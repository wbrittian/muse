#pragma once

#include <cmath>
#include <Eigen/Dense>

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowVector = Eigen::RowVectorXf;

struct LayerNorm {
    RowVector weight;
    RowVector bias;
    float eps = 1e-5f;

    LayerNorm() = default;
    LayerNorm(int d_model)
    : weight(RowVector::Ones(d_model))
    , bias(RowVector::Zero(d_model))
    {}

    void forward(Matrix& x) const {
        for (int i = 0; i < x.rows(); i++) {
            auto row = x.row(i);
            row.array() -= row.mean();
            float var = row.squaredNorm() / row.size();
            row *= 1.0f / std::sqrt(var + eps);
            row = row.cwiseProduct(weight) + bias;
        }
    }
};

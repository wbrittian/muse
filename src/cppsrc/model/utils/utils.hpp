#pragma once

#include <Eigen/Dense>

struct LayerNorm {
    Eigen::VectorXd weight;
    Eigen::VectorXd bias;
    double eps = 1e-5;

    LayerNorm() = default;
    LayerNorm(int d_model)
    : weight(Eigen::VectorXd::Ones(d_model))
    , bias(Eigen::VectorXd::Zero(d_model))
    {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) {
        Eigen::VectorXd mean = x.rowwise().mean();
        Eigen::MatrixXd centered = x.colwise() - mean;
        Eigen::VectorXd var = centered.array().square().rowwise().mean();
        Eigen::MatrixXd normed = centered.array().colwise() / (var.array() + eps).sqrt();
        return (normed.array().rowwise() * weight.transpose().array()).rowwise() + bias.transpose().array();
    }
};
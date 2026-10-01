#pragma once

#include <Eigen/Dense>

class FeedForward {
public:
    Eigen::MatrixXd W1, W2;
    Eigen::VectorXd b1, b2;

    FeedForward() = default;
    FeedForward(int d_model, int dim_ff)
    : W1(Eigen::MatrixXd::Zero(dim_ff, d_model))
    , W2(Eigen::MatrixXd::Zero(d_model, dim_ff))
    , b1(Eigen::VectorXd::Zero(dim_ff))
    , b2(Eigen::VectorXd::Zero(d_model))
    {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x);

private:
    Eigen::MatrixXd gelu(const Eigen::MatrixXd& x);
};
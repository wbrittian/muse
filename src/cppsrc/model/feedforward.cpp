#include <cmath>

#include "feedforward.hpp"

Eigen::MatrixXd FeedForward::gelu(const Eigen::MatrixXd& x) {
    return x.array() * 0.5 * (1.0 + (x.array() / std::sqrt(2.0)).unaryExpr([](double v) {
        return std::erf(v);
    }));
}

Eigen::MatrixXd FeedForward::forward(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd h = (x * W1.transpose()).rowwise() + b1.transpose();
    h = gelu(h);
    return (h * W2.transpose()).rowwise() + b2.transpose();
}
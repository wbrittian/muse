#pragma once

#include <Eigen/Dense>
#include <math.h>
#include <vector>

class MultiHeadAttention {
public:
    MultiHeadAttention(const int& num_heads)
    : num_heads(num_heads)
    {}

private:
    int num_heads;

    Eigen::MatrixXd softmax(const Eigen::MatrixXd& matrix);
    Eigen::MatrixXd attend(
        const Eigen::MatrixXd& keys,
        const Eigen::MatrixXd& queries,
        const Eigen::MatrixXd& values,
        const Eigen::MatrixXd& mask
    );
    std::vector<Eigen::MatrixXd> batch(const Eigen::MatrixXd& matrix);

    Eigen::MatrixXd forward();
};
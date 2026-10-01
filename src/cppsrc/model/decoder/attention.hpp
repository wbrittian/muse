#pragma once

#include <Eigen/Dense>
#include <vector>

class MultiHeadAttention {
public:
    // W_in is (3*d_model, d_model) — Q, K, V projections concatenated, matching PyTorch's in_proj_weight
    Eigen::MatrixXd W_in;
    Eigen::VectorXd b_in;
    Eigen::MatrixXd W_out;
    Eigen::VectorXd b_out;

    MultiHeadAttention() = default;
    MultiHeadAttention(int d_model, int num_heads)
    : num_heads(num_heads)
    , d_model(d_model)
    , d_head(d_model / num_heads)
    , W_in(Eigen::MatrixXd::Zero(3 * d_model, d_model))
    , b_in(Eigen::VectorXd::Zero(3 * d_model))
    , W_out(Eigen::MatrixXd::Zero(d_model, d_model))
    , b_out(Eigen::VectorXd::Zero(d_model))
    {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& mask);

private:
    int num_heads, d_model, d_head;

    Eigen::MatrixXd softmax(const Eigen::MatrixXd& matrix);
    Eigen::MatrixXd attend(
        const Eigen::MatrixXd& keys,
        const Eigen::MatrixXd& queries,
        const Eigen::MatrixXd& values,
        const Eigen::MatrixXd& mask
    );
    std::vector<Eigen::MatrixXd> batch(const Eigen::MatrixXd& matrix);
};
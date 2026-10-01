#include <cmath>

#include "attention.hpp"

Eigen::MatrixXd MultiHeadAttention::softmax(const Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd result(matrix.rows(), matrix.cols());
    for (int i = 0; i < matrix.rows(); i++) {
        // subtract max for numerical stability — exp(-inf) = 0 so masked positions zero out correctly
        double max_val = matrix.row(i).maxCoeff();
        Eigen::VectorXd exp_row = (matrix.row(i).array() - max_val).exp();
        result.row(i) = exp_row / exp_row.sum();
    }
    return result;
}

Eigen::MatrixXd MultiHeadAttention::attend(
    const Eigen::MatrixXd& keys,
    const Eigen::MatrixXd& queries,
    const Eigen::MatrixXd& values,
    const Eigen::MatrixXd& mask
) {
    // scale by sqrt(d_head), not sqrt(seq_len)
    Eigen::MatrixXd S = (queries * keys.transpose()) / std::sqrt((double)queries.cols());
    S += mask;
    return softmax(S) * values;
}

std::vector<Eigen::MatrixXd> MultiHeadAttention::batch(const Eigen::MatrixXd& matrix) {
    std::vector<Eigen::MatrixXd> heads;
    for (int h = 0; h < num_heads; h++) {
        heads.push_back(matrix.block(0, h * d_head, matrix.rows(), d_head));
    }
    return heads;
}

Eigen::MatrixXd MultiHeadAttention::forward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& mask) {
    Eigen::MatrixXd qkv = (x * W_in.transpose()).rowwise() + b_in.transpose();

    auto Q_heads = batch(qkv.leftCols(d_model));
    auto K_heads = batch(qkv.middleCols(d_model, d_model));
    auto V_heads = batch(qkv.rightCols(d_model));

    Eigen::MatrixXd concat(x.rows(), d_model);
    for (int h = 0; h < num_heads; h++) {
        concat.block(0, h * d_head, x.rows(), d_head) = attend(K_heads[h], Q_heads[h], V_heads[h], mask);
    }

    return (concat * W_out.transpose()).rowwise() + b_out.transpose();
}
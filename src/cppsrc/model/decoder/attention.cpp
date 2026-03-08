#include "attention.hpp"

Eigen::MatrixXd softmax(const Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd result(matrix.rows(), matrix.cols());
    for (int i = 0; i < matrix.rows(); i++) {
        Eigen::VectorXd exp_row = matrix.row(i).array().exp();
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
    int d = queries.rows();

    Eigen::MatrixXd S = (queries * keys.transpose()) / sqrt(d);
    S += mask;
    S = softmax(S);
    return S * values;
}
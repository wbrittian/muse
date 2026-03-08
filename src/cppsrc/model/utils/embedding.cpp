#include "embedding.hpp"


Eigen::MatrixXd Embedding::embed(const std::vector<int>& tokens) {
    Eigen::MatrixXd result(tokens.size(), d_model);
    for (int i = 0; i < tokens.size(); i++) {
        Eigen::RowVectorXd token_embed = embedding.row(tokens[i]);
        result.row(i) = token_embed;
    }

    return result;
}

void Embedding::set_embedding(const Eigen::MatrixXd& e) {
    embedding = e;
}

void Embedding::apply_gradient(const Eigen::MatrixXd& gradient, const float& lr) {
    embedding -= lr * gradient;
}

Eigen::MatrixXd Embedding::get_embedding() {
    return embedding;
}
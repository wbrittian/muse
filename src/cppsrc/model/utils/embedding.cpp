#include "embedding.hpp"


Eigen::MatrixXd Embedding::embed(const std::vector<int>& tokens) {
    Eigen::MatrixXd result(tokens.size(), d_model);
    for (int i = 0; i < tokens.size(); i++) {
        Eigen::RowVectorXd token_embed = embedding.row(tokens[i]);
        result.row(i) = token_embed;
    }

    return result;
}

void Embedding::update_embedding(const Eigen::MatrixXd& new_embedding) {
    embedding = new_embedding;
}

Eigen::MatrixXd Embedding::get_embedding() {
    return embedding;
}
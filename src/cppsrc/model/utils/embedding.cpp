#include <stdexcept>
#include <string>

#include "embedding.hpp"

Matrix Embedding::embed(const std::vector<int>& tokens) const {
    Matrix result(tokens.size(), d_model);
    for (int i = 0; i < (int)tokens.size(); i++) {
        if (tokens[i] < 0 || tokens[i] >= embedding.rows())
            throw std::out_of_range("token id out of range: " + std::to_string(tokens[i]));
        result.row(i) = embedding.row(tokens[i]);
    }
    return result;
}

void Embedding::set_embedding(const Matrix& e) {
    embedding = e;
}

const Matrix& Embedding::get_embedding() const {
    return embedding;
}

#pragma once

#include <vector>

#include "utils.hpp"

class Embedding {
public:
    Embedding(int vocab_size, int d_model)
    : d_model(d_model)
    , embedding(Matrix::Zero(vocab_size, d_model))
    {}

    Matrix embed(const std::vector<int>& tokens) const;

    void set_embedding(const Matrix& e);
    const Matrix& get_embedding() const;

private:
    int d_model;
    Matrix embedding;
};

#pragma once

#include "../utils/utils.hpp"

struct KVCache {
    Matrix keys;
    Matrix values;
};

class MultiHeadAttention {
public:
    // W_in is (3*d_model, d_model) — Q, K, V projections concatenated, matching PyTorch's in_proj_weight
    Matrix W_in;
    RowVector b_in;
    Matrix W_out;
    RowVector b_out;

    MultiHeadAttention() = default;
    MultiHeadAttention(int d_model, int num_heads)
    : W_in(Matrix::Zero(3 * d_model, d_model))
    , b_in(RowVector::Zero(3 * d_model))
    , W_out(Matrix::Zero(d_model, d_model))
    , b_out(RowVector::Zero(d_model))
    , num_heads(num_heads)
    , d_model(d_model)
    , d_head(d_model / num_heads)
    {}

    // x holds positions [pos, pos + x.rows()); their keys and values are appended to the cache
    Matrix forward(const Matrix& x, KVCache& cache, int pos) const;

private:
    int num_heads, d_model, d_head;
};

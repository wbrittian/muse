#include <cmath>

#include "attention.hpp"

Matrix MultiHeadAttention::forward(const Matrix& x, KVCache& cache, int pos) const {
    int n = x.rows();
    int len = pos + n;

    Matrix qkv = (x * W_in.transpose()).rowwise() + b_in;
    cache.keys.middleRows(pos, n) = qkv.middleCols(d_model, d_model);
    cache.values.middleRows(pos, n) = qkv.rightCols(d_model);

    float scale = 1.0f / std::sqrt((float)d_head);
    Matrix scores(n, len);
    Matrix concat(n, d_model);
    for (int h = 0; h < num_heads; h++) {
        scores.noalias() = qkv.middleCols(h * d_head, d_head)
                         * cache.keys.block(0, h * d_head, len, d_head).transpose();
        scores *= scale;

        for (int i = 0; i < n; i++) {
            int visible = pos + i + 1;
            auto row = scores.row(i).head(visible);
            row = (row.array() - row.maxCoeff()).exp();
            row /= row.sum();
            scores.row(i).tail(len - visible).setZero();
        }

        concat.middleCols(h * d_head, d_head).noalias() = scores * cache.values.block(0, h * d_head, len, d_head);
    }

    return (concat * W_out.transpose()).rowwise() + b_out;
}

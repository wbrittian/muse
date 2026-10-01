#pragma once

#include "utils/utils.hpp"

class FeedForward {
public:
    Matrix W1, W2;
    RowVector b1, b2;

    FeedForward() = default;
    FeedForward(int d_model, int dim_ff)
    : W1(Matrix::Zero(dim_ff, d_model))
    , W2(Matrix::Zero(d_model, dim_ff))
    , b1(RowVector::Zero(dim_ff))
    , b2(RowVector::Zero(d_model))
    {}

    Matrix forward(const Matrix& x) const;
};

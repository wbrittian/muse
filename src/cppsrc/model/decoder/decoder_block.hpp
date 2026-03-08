#pragma once

#include <Eigen/Dense>

#include "attention.hpp"
#include "../feedforward.hpp"
#include "../utils/utils.hpp"

class DecoderBlock {
public:
    MultiHeadAttention attention;
    FeedForward ff;
    LayerNorm norm1, norm2;

    DecoderBlock() = default;
    DecoderBlock(int d_model, int num_heads, int dim_ff)
    : attention(d_model, num_heads)
    , ff(d_model, dim_ff)
    , norm1(d_model)
    , norm2(d_model)
    {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& mask);
};
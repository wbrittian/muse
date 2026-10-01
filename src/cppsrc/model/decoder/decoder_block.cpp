#include "decoder_block.hpp"

Eigen::MatrixXd DecoderBlock::forward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& mask) {
    Eigen::MatrixXd h = norm1.forward(x + attention.forward(x, mask));
    return norm2.forward(h + ff.forward(h));
}
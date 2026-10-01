#include "decoder_block.hpp"

Matrix DecoderBlock::forward(const Matrix& x, KVCache& cache, int pos) const {
    Matrix h = x + attention.forward(x, cache, pos);
    norm1.forward(h);
    Matrix out = h + ff.forward(h);
    norm2.forward(out);
    return out;
}

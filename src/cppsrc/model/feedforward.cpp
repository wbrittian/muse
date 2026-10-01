#include <cmath>

#include "feedforward.hpp"

Matrix FeedForward::forward(const Matrix& x) const {
    Matrix h = (x * W1.transpose()).rowwise() + b1;
    h = h.unaryExpr([](float v) { return 0.5f * v * (1.0f + std::erf(v * float(M_SQRT1_2))); });
    return (h * W2.transpose()).rowwise() + b2;
}

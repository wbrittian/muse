#pragma once

#include <Eigen/Dense>

class DecoderBlock {
public:
    Eigen::MatrixXd forward(Eigen::MatrixXd embedding);

private:
};
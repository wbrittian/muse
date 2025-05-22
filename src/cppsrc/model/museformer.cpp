#include "museformer.hpp"
#include <math.h>

// public
void Museformer::loadWeights(const std::string& path) {

}

// private
void Museformer::init_positional_encoding() {
    for (int i; i < max_seq_len; i++) {
        for (int j; j < (d_model / 2) - 1; j++) {
            positional_encoding(i, 2 * j) = sin(i / pow(10000, ((2 * j) / d_model)));
            positional_encoding(i, 2 * j + 1) = cos(i / pow(10000, ((2 * j) / d_model)));
        }
    }
}

Eigen::MatrixXd Museformer::forward(const std::vector<int>& input_tokens) {
    embedding.embed(input_tokens);

    
}

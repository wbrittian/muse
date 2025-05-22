#pragma once

#include <string>
#include <Eigen/Dense>

#include "utils/embedding.hpp"

class Museformer {
public:
    // constructor
    Museformer(
        int vocab_size,
        int max_seq_len,
        int d_model,
        int num_heads,
        int num_layers,
        int dim_ff,
        float p_drop
    )
    : vocab_size(vocab_size)
    , max_seq_len(max_seq_len)
    , d_model(d_model)
    , num_heads(num_heads)
    , num_layers(num_layers)
    , dim_ff(dim_ff)
    , p_drop(p_drop)
    
    , embedding(vocab_size, d_model)
    {}

    void loadWeights(const std::string& path) {

    }

    Eigen::MatrixXd forward() {

    }

private:
    // hyperparams
    int vocab_size;
    int max_seq_len;
    int d_model;
    int num_heads;
    int num_layers;
    int dim_ff;
    float p_drop;

    Embedding embedding;
};
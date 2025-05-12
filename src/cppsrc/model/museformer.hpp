#pragma once

#include <Eigen/Dense>

class Museformer {
public:
    void init(
        int vocab_size,
        int max_seq_len,
        int d_model,
        int num_heads,
        int num_layers,
        int dim_ff,
        int p_drop
    );

    void test() {
        std::cout << "hello" << std::endl;
    }

private:
    // hyperparams
    int vocab_size;
    int max_seq_len;
    int d_model;
    int num_heads;
    int num_layers;
    int dim_ff;
    int p_drop;


};
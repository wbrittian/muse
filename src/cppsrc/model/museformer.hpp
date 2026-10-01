#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

#include "utils/embedding.hpp"
#include "decoder/decoder_block.hpp"

class Museformer {
public:
    Museformer(
        int vocab_size,
        int max_seq_len,
        int d_model,
        int num_heads,
        int num_layers,
        int dim_ff
    )
    : vocab_size(vocab_size)
    , max_seq_len(max_seq_len)
    , d_model(d_model)
    , num_heads(num_heads)
    , num_layers(num_layers)
    , dim_ff(dim_ff)
    , embedding(vocab_size, d_model)
    , positional_encoding(Eigen::MatrixXd::Zero(max_seq_len, d_model))
    , W_proj(Eigen::MatrixXd::Zero(vocab_size, d_model))
    , b_proj(Eigen::VectorXd::Zero(vocab_size))
    {
        for (int i = 0; i < num_layers; i++) {
            decoder_blocks.emplace_back(d_model, num_heads, dim_ff);
        }
    }

    void save(const std::string& path);
    void load(const std::string& path);

    std::vector<int> generate(const std::vector<int>& input_tokens);

private:
    int vocab_size, max_seq_len, d_model, num_heads, num_layers, dim_ff;

    Embedding embedding;
    Eigen::MatrixXd positional_encoding;
    std::vector<DecoderBlock> decoder_blocks;
    Eigen::MatrixXd W_proj;
    Eigen::VectorXd b_proj;

    Eigen::MatrixXd causal_mask(int seq_len);
    Eigen::MatrixXd forward(const std::vector<int>& input_tokens);
};
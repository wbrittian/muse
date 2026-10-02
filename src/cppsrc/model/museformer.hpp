#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

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
    , positional_encoding(Matrix::Zero(max_seq_len, d_model))
    , W_proj(Matrix::Zero(vocab_size, d_model))
    , b_proj(RowVector::Zero(vocab_size))
    , rng(std::random_device{}())
    {
        for (int i = 0; i < num_layers; i++) {
            decoder_blocks.emplace_back(d_model, num_heads, dim_ff);
            caches.push_back({Matrix(max_seq_len, d_model), Matrix(max_seq_len, d_model)});
        }
    }

    void save(const std::string& path) const;
    void load(const std::string& path);

    Matrix forward(const std::vector<int>& input_tokens);

    // top-k sampling over allowed_tokens (all if empty); stops before eos_token or at max_len.
    // Optional length guard: rest_divs/note_divs give each token's length in divisions
    // (a NOTE's counts once the next token, its PITCH, arrives). eos_token is masked
    // while elapsed < eos_after_divs, and generation stops once elapsed >= stop_at_divs
    // (either < 0 disables it).
    std::vector<int> generate(
        const std::vector<int>& input_tokens,
        int max_len,
        int top_k = 8,
        float temperature = 1.0f,
        const std::vector<int>& allowed_tokens = {},
        std::optional<uint64_t> seed = std::nullopt,
        int eos_token = 1,
        const std::vector<int>& rest_divs = {},
        const std::vector<int>& note_divs = {},
        int eos_after_divs = -1,
        int stop_at_divs = -1
    );

private:
    int vocab_size, max_seq_len, d_model, num_heads, num_layers, dim_ff;

    Embedding embedding;
    Matrix positional_encoding;
    std::vector<DecoderBlock> decoder_blocks;
    Matrix W_proj;
    RowVector b_proj;

    std::vector<KVCache> caches;
    std::mt19937_64 rng;

    Matrix decode(const std::vector<int>& tokens, int pos);
    int sample(const RowVector& logits, int top_k, float temperature, const std::vector<int>& candidates);
};

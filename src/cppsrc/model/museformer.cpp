#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "museformer.hpp"

// binary I/O helpers — row-major float64 with int32 shape headers, as written by export_weights.py

using FileMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static void write_matrix(std::ofstream& f, const Matrix& m) {
    int32_t rows = m.rows(), cols = m.cols();
    FileMatrix d = m.cast<double>();
    f.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(double));
}

static void write_vector(std::ofstream& f, const RowVector& v) {
    int32_t n = v.size();
    Eigen::RowVectorXd d = v.cast<double>();
    f.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(double));
}

static Matrix read_matrix(std::ifstream& f, int rows, int cols, const std::string& name) {
    int32_t r = 0, c = 0;
    f.read(reinterpret_cast<char*>(&r), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&c), sizeof(int32_t));
    if (!f || r != rows || c != cols)
        throw std::runtime_error(name + ": expected " + std::to_string(rows) + "x" + std::to_string(cols)
                                 + ", got " + std::to_string(r) + "x" + std::to_string(c));
    FileMatrix m(r, c);
    f.read(reinterpret_cast<char*>(m.data()), m.size() * sizeof(double));
    if (!f) throw std::runtime_error(name + ": unexpected end of file");
    return m.cast<float>();
}

static RowVector read_vector(std::ifstream& f, int n, const std::string& name) {
    int32_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
    if (!f || len != n)
        throw std::runtime_error(name + ": expected length " + std::to_string(n) + ", got " + std::to_string(len));
    Eigen::RowVectorXd v(len);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(double));
    if (!f) throw std::runtime_error(name + ": unexpected end of file");
    return v.cast<float>();
}

// public

void Museformer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);

    write_matrix(f, embedding.get_embedding());
    write_matrix(f, positional_encoding);

    for (auto& block : decoder_blocks) {
        write_matrix(f, block.attention.W_in);
        write_vector(f, block.attention.b_in);
        write_matrix(f, block.attention.W_out);
        write_vector(f, block.attention.b_out);
        write_matrix(f, block.ff.W1);
        write_vector(f, block.ff.b1);
        write_matrix(f, block.ff.W2);
        write_vector(f, block.ff.b2);
        write_vector(f, block.norm1.weight);
        write_vector(f, block.norm1.bias);
        write_vector(f, block.norm2.weight);
        write_vector(f, block.norm2.bias);
    }

    write_matrix(f, W_proj);
    write_vector(f, b_proj);
}

void Museformer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);

    embedding.set_embedding(read_matrix(f, vocab_size, d_model, "token_embed"));
    positional_encoding = read_matrix(f, max_seq_len, d_model, "pos_embed");

    for (int i = 0; i < num_layers; i++) {
        auto& block = decoder_blocks[i];
        std::string p = "layer " + std::to_string(i) + " ";
        block.attention.W_in  = read_matrix(f, 3 * d_model, d_model, p + "in_proj weight");
        block.attention.b_in  = read_vector(f, 3 * d_model, p + "in_proj bias");
        block.attention.W_out = read_matrix(f, d_model, d_model, p + "out_proj weight");
        block.attention.b_out = read_vector(f, d_model, p + "out_proj bias");
        block.ff.W1           = read_matrix(f, dim_ff, d_model, p + "linear1 weight");
        block.ff.b1           = read_vector(f, dim_ff, p + "linear1 bias");
        block.ff.W2           = read_matrix(f, d_model, dim_ff, p + "linear2 weight");
        block.ff.b2           = read_vector(f, d_model, p + "linear2 bias");
        block.norm1.weight    = read_vector(f, d_model, p + "norm1 weight");
        block.norm1.bias      = read_vector(f, d_model, p + "norm1 bias");
        block.norm2.weight    = read_vector(f, d_model, p + "norm2 weight");
        block.norm2.bias      = read_vector(f, d_model, p + "norm2 bias");
    }

    W_proj = read_matrix(f, vocab_size, d_model, "output_proj weight");
    b_proj = read_vector(f, vocab_size, "output_proj bias");

    if (f.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("trailing data in " + path + " (num_layers mismatch?)");
}

Matrix Museformer::forward(const std::vector<int>& input_tokens) {
    Matrix x = decode(input_tokens, 0);
    return (x * W_proj.transpose()).rowwise() + b_proj;
}

std::vector<int> Museformer::generate(
    const std::vector<int>& input_tokens,
    int max_len,
    int top_k,
    float temperature,
    const std::vector<int>& allowed_tokens,
    std::optional<uint64_t> seed,
    int eos_token,
    const std::vector<int>& rest_divs,
    const std::vector<int>& note_divs,
    int eos_after_divs,
    int stop_at_divs
) {
    if (input_tokens.empty()) throw std::invalid_argument("input_tokens is empty");
    if (top_k < 1) throw std::invalid_argument("top_k must be at least 1");
    if (!(temperature > 0)) throw std::invalid_argument("temperature must be positive");
    if (seed) rng.seed(*seed);
    max_len = std::min(max_len, max_seq_len);

    std::vector<int> candidates = allowed_tokens;
    if (candidates.empty()) {
        candidates.resize(vocab_size);
        std::iota(candidates.begin(), candidates.end(), 0);
    }
    for (int t : candidates)
        if (t < 0 || t >= vocab_size) throw std::out_of_range("allowed token out of range: " + std::to_string(t));

    bool guarded = eos_after_divs >= 0 || stop_at_divs >= 0;
    if (guarded && ((int)rest_divs.size() != vocab_size || (int)note_divs.size() != vocab_size))
        throw std::invalid_argument("rest_divs and note_divs must have vocab_size entries");
    std::vector<int> no_eos;
    for (int t : candidates)
        if (t != eos_token) no_eos.push_back(t);
    if (eos_after_divs >= 0 && no_eos.empty())
        throw std::invalid_argument("allowed tokens hold nothing but eos_token");

    std::vector<int> output = input_tokens;
    if ((int)output.size() >= max_len) return output;

    int elapsed = 0, pending = 0;
    Matrix x = decode(output, 0);
    RowVector logits = (x.bottomRows(1) * W_proj.transpose()) + b_proj;
    while ((int)output.size() < max_len) {
        if (stop_at_divs >= 0 && elapsed >= stop_at_divs) break;
        const auto& pool = (eos_after_divs >= 0 && elapsed < eos_after_divs) ? no_eos : candidates;

        int next_token = sample(logits, top_k, temperature, pool);
        if (next_token == eos_token) break;
        output.push_back(next_token);
        if (guarded) {
            if (rest_divs[next_token]) {
                elapsed += rest_divs[next_token];
            } else if (note_divs[next_token]) {
                pending = note_divs[next_token];
            } else {
                elapsed += pending;
                pending = 0;
            }
        }
        if ((int)output.size() == max_len) break;

        x = decode({next_token}, output.size() - 1);
        logits = (x * W_proj.transpose()) + b_proj;
    }

    return output;
}

// private

Matrix Museformer::decode(const std::vector<int>& tokens, int pos) {
    if (pos + (int)tokens.size() > max_seq_len)
        throw std::length_error("sequence longer than max_seq_len (" + std::to_string(max_seq_len) + ")");

    Matrix x = embedding.embed(tokens);
    x += positional_encoding.middleRows(pos, tokens.size());

    for (int i = 0; i < num_layers; i++) {
        x = decoder_blocks[i].forward(x, caches[i], pos);
    }
    return x;
}

int Museformer::sample(const RowVector& logits, int top_k, float temperature, const std::vector<int>& candidates) {
    std::vector<int> ids = candidates;
    int k = std::min<int>(top_k, ids.size());
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                      [&](int a, int b) { return logits(a) > logits(b); });

    double top = logits(ids[0]) / temperature;
    std::vector<double> weights(k);
    for (int i = 0; i < k; i++) {
        weights[i] = std::exp(logits(ids[i]) / temperature - top);
    }
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return ids[dist(rng)];
}

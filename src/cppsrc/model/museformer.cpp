#include <fstream>
#include <limits>
#include <stdexcept>

#include "museformer.hpp"

// binary I/O helpers — all weights stored row-major float64

static void write_matrix(std::ofstream& f, const Eigen::MatrixXd& m) {
    int rows = m.rows(), cols = m.cols();
    f.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    f.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            double v = m(i, j);
            f.write(reinterpret_cast<const char*>(&v), sizeof(double));
        }
}

static void write_vector(std::ofstream& f, const Eigen::VectorXd& v) {
    int n = v.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(int));
    f.write(reinterpret_cast<const char*>(v.data()), n * sizeof(double));
}

static Eigen::MatrixXd read_matrix(std::ifstream& f) {
    int rows, cols;
    f.read(reinterpret_cast<char*>(&rows), sizeof(int));
    f.read(reinterpret_cast<char*>(&cols), sizeof(int));
    Eigen::MatrixXd m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            f.read(reinterpret_cast<char*>(&m(i, j)), sizeof(double));
    return m;
}

static Eigen::VectorXd read_vector(std::ifstream& f) {
    int n;
    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    Eigen::VectorXd v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(double));
    return v;
}

// public

void Museformer::save(const std::string& path) {
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

    embedding.set_embedding(read_matrix(f));
    positional_encoding = read_matrix(f);

    for (auto& block : decoder_blocks) {
        block.attention.W_in  = read_matrix(f);
        block.attention.b_in  = read_vector(f);
        block.attention.W_out = read_matrix(f);
        block.attention.b_out = read_vector(f);
        block.ff.W1           = read_matrix(f);
        block.ff.b1           = read_vector(f);
        block.ff.W2           = read_matrix(f);
        block.ff.b2           = read_vector(f);
        block.norm1.weight    = read_vector(f);
        block.norm1.bias      = read_vector(f);
        block.norm2.weight    = read_vector(f);
        block.norm2.bias      = read_vector(f);
    }

    W_proj = read_matrix(f);
    b_proj = read_vector(f);
}

// private

Eigen::MatrixXd Museformer::causal_mask(int seq_len) {
    Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(seq_len, seq_len);
    for (int i = 0; i < seq_len; i++)
        for (int j = i + 1; j < seq_len; j++)
            mask(i, j) = -std::numeric_limits<double>::infinity();
    return mask;
}

Eigen::MatrixXd Museformer::forward(const std::vector<int>& input_tokens) {
    Eigen::MatrixXd x = embedding.embed(input_tokens);
    x += positional_encoding.topRows(input_tokens.size());

    Eigen::MatrixXd mask = causal_mask(input_tokens.size());
    for (auto& block : decoder_blocks) {
        x = block.forward(x, mask);
    }

    return (x * W_proj.transpose()).rowwise() + b_proj.transpose();
}

std::vector<int> Museformer::generate(const std::vector<int>& input_tokens) {
    std::vector<int> output = input_tokens;

    int max_new = max_seq_len - (int)input_tokens.size();
    for (int i = 0; i < max_new; i++) {
        Eigen::MatrixXd logits = forward(output);

        int next_token;
        logits.row(logits.rows() - 1).maxCoeff(&next_token);
        output.push_back(next_token);

        if (next_token == 1) break;  // EOS
    }

    return output;
}
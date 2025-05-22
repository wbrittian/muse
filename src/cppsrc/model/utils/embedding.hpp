#pragma once

#include <Eigen/Dense>
#include <random>
#include <vector>

class Embedding {
public:
    Embedding(int vocab_size, int d_model)
    : d_model(d_model)
    , embedding(vocab_size, d_model)
    {
        float stdev = std::sqrt(2.0f / (vocab_size + d_model));
        std::mt19937 gen {std::random_device{}()};
        std::normal_distribution<double> dist(0.0f, stdev);

        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < d_model; j++) {
                embedding(i, j) = dist(gen);
            }
        }
    }

    Eigen::MatrixXd embed(const std::vector<int>& tokens);

    void update_embedding(const Eigen::MatrixXd& new_embedding);
    Eigen::MatrixXd get_embedding();

private:
    int d_model;
    Eigen::MatrixXd embedding;
};
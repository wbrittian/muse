#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>

#include <gtest/gtest.h>

#include "cppsrc/model/museformer.hpp"

namespace {

constexpr int VOCAB = 20, MAX_LEN = 16, D_MODEL = 8, HEADS = 2, LAYERS = 2, DIM_FF = 12;

void write_random(std::ofstream& f, std::mt19937& gen, std::vector<int32_t> shape) {
    std::normal_distribution<double> dist(0.0, 0.5);
    f.write(reinterpret_cast<const char*>(shape.data()), shape.size() * sizeof(int32_t));
    int n = 1;
    for (int s : shape) n *= s;
    for (int i = 0; i < n; i++) {
        double v = dist(gen);
        f.write(reinterpret_cast<const char*>(&v), sizeof(double));
    }
}

std::string random_weights(int num_layers = LAYERS, int max_len = MAX_LEN) {
    std::string path = (std::filesystem::temp_directory_path() / "museformer_test.bin").string();
    std::ofstream f(path, std::ios::binary);
    std::mt19937 gen(42);
    write_random(f, gen, {VOCAB, D_MODEL});
    write_random(f, gen, {max_len, D_MODEL});
    for (int i = 0; i < num_layers; i++) {
        write_random(f, gen, {3 * D_MODEL, D_MODEL});
        write_random(f, gen, {3 * D_MODEL});
        write_random(f, gen, {D_MODEL, D_MODEL});
        write_random(f, gen, {D_MODEL});
        write_random(f, gen, {DIM_FF, D_MODEL});
        write_random(f, gen, {DIM_FF});
        write_random(f, gen, {D_MODEL, DIM_FF});
        write_random(f, gen, {D_MODEL});
        for (int j = 0; j < 4; j++) write_random(f, gen, {D_MODEL});
    }
    write_random(f, gen, {VOCAB, D_MODEL});
    write_random(f, gen, {VOCAB});
    return path;
}

Museformer small_model() {
    return Museformer(VOCAB, MAX_LEN, D_MODEL, HEADS, LAYERS, DIM_FF);
}

}

TEST(LayerNormTest, NormalizesRows) {
    LayerNorm norm(4);
    norm.weight << 1, 2, 1, 1;
    norm.bias << 0, 0, 0, 1;
    Matrix x(1, 4);
    x << 1, 2, 3, 4;
    norm.forward(x);

    double sd = std::sqrt(1.25 + 1e-5);
    EXPECT_NEAR(x(0, 0), -1.5 / sd, 1e-6);
    EXPECT_NEAR(x(0, 1), 2 * -0.5 / sd, 1e-6);
    EXPECT_NEAR(x(0, 2), 0.5 / sd, 1e-6);
    EXPECT_NEAR(x(0, 3), 1.5 / sd + 1, 1e-6);
}

TEST(FeedForwardTest, UsesExactGelu) {
    FeedForward ff(1, 1);
    ff.W1 << 1;
    ff.W2 << 1;
    Matrix x(2, 1);
    x << 1, -2;
    Matrix y = ff.forward(x);

    EXPECT_NEAR(y(0, 0), 0.8413447460685429, 1e-6);
    EXPECT_NEAR(y(1, 0), -0.04550026389635842, 1e-6);
}

TEST(MuseformerTest, SaveLoadRoundTrip) {
    Museformer a = small_model(), b = small_model();
    a.load(random_weights());
    std::string path = (std::filesystem::temp_directory_path() / "museformer_roundtrip.bin").string();
    a.save(path);
    b.load(path);

    std::vector<int> tokens = {0, 3, 7, 1, 19};
    EXPECT_TRUE(a.forward(tokens).isApprox(b.forward(tokens)));
}

TEST(MuseformerTest, LoadRejectsMismatchedShapes) {
    Museformer model = small_model();
    EXPECT_THROW(model.load(random_weights(LAYERS, MAX_LEN + 1)), std::runtime_error);
    EXPECT_THROW(model.load(random_weights(LAYERS + 1)), std::runtime_error);
    EXPECT_THROW(model.load(random_weights(LAYERS - 1)), std::runtime_error);
    EXPECT_THROW(model.load("does/not/exist.bin"), std::runtime_error);
}

TEST(MuseformerTest, IsCausal) {
    Museformer model = small_model();
    model.load(random_weights());

    Matrix a = model.forward({0, 4, 5, 6, 7});
    Matrix b = model.forward({0, 4, 5, 9, 2});
    EXPECT_TRUE(a.topRows(3).isApprox(b.topRows(3)));
    EXPECT_FALSE(a.row(3).isApprox(b.row(3)));
}

TEST(MuseformerTest, RejectsBadInput) {
    Museformer model = small_model();
    model.load(random_weights());

    EXPECT_THROW(model.forward({0, VOCAB}), std::out_of_range);
    EXPECT_THROW(model.forward(std::vector<int>(MAX_LEN + 1, 0)), std::length_error);
    EXPECT_THROW(model.generate({}, MAX_LEN), std::invalid_argument);
    EXPECT_THROW(model.generate({0}, MAX_LEN, 0), std::invalid_argument);
    EXPECT_THROW(model.generate({0}, MAX_LEN, 8, 1.0f, {VOCAB}), std::out_of_range);
}

TEST(MuseformerTest, GreedyGenerateMatchesFullForward) {
    Museformer model = small_model();
    model.load(random_weights());

    std::vector<int> output = model.generate({0, 3}, MAX_LEN, 1, 1.0f, {}, std::nullopt, -1);
    ASSERT_EQ(output.size(), MAX_LEN);

    Matrix logits = model.forward(output);
    for (int i = 1; i + 1 < MAX_LEN; i++) {
        int best;
        logits.row(i).maxCoeff(&best);
        EXPECT_EQ(output[i + 1], best) << "position " << i + 1;
    }
}

TEST(MuseformerTest, SamplesOnlyAllowedTopK) {
    Museformer model = small_model();
    model.load(random_weights());

    std::vector<int> allowed = {2, 5, 11, 13, 17};
    for (uint64_t seed = 0; seed < 20; seed++) {
        std::vector<int> output = model.generate({0}, MAX_LEN, 2, 1.0f, allowed, seed, -1);
        Matrix logits = model.forward(output);
        for (int i = 1; i < (int)output.size(); i++) {
            std::vector<int> ranked = allowed;
            std::sort(ranked.begin(), ranked.end(),
                      [&](int a, int b) { return logits(i - 1, a) > logits(i - 1, b); });
            EXPECT_TRUE(output[i] == ranked[0] || output[i] == ranked[1]);
        }
    }
}

TEST(MuseformerTest, SeedIsReproducibleAndStopsAtEos) {
    Museformer model = small_model();
    model.load(random_weights());

    auto a = model.generate({0}, MAX_LEN, 5, 1.0f, {}, 7);
    auto b = model.generate({0}, MAX_LEN, 5, 1.0f, {}, 7);
    EXPECT_EQ(a, b);
    for (int t : a) EXPECT_NE(t, 1);

    EXPECT_EQ(model.generate({0, 1, 2}, 3), (std::vector<int>{0, 1, 2}));
}

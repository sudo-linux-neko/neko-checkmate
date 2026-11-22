// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include "training.h"
#include "board.h"
#include "evaluate.h"
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <filesystem>

namespace neko {

static std::mt19937 train_rng(42);

void OptimizerState::init(size_t num_params) {
    m.assign(num_params, 0.0f);
    v.assign(num_params, 0.0f);
    timestep = 0;
}

bool OptimizerState::load(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) return false;
    f.read(reinterpret_cast<char*>(&timestep), sizeof(timestep));
    size_t sz = 0;
    f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    m.resize(sz);
    v.resize(sz);
    f.read(reinterpret_cast<char*>(m.data()), sz * sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), sz * sizeof(float));
    return f.good();
}

void OptimizerState::save(const std::string& filename) const {
    std::ofstream f(filename, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&timestep), sizeof(timestep));
    size_t sz = m.size();
    f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    f.write(reinterpret_cast<const char*>(m.data()), sz * sizeof(float));
    f.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(float));
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float sigmoid_derivative(float s) {
    return s * (1.0f - s);
}

void pos_to_features(const Board& board, std::vector<float>& features) {
    features.assign(NNUE_INPUT, 0.0f);
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.piece_on(sq);
        if (p != NO_PIECE) {
            int idx = feature_index(p, sq);
            features[idx] = 1.0f;
        }
    }
}

// Single quantization scale — must match Q in nnue.cpp exactly.
static constexpr int TRAIN_Q = 128;

// SGD training with binary cross-entropy loss.
// Forward pass and gradient scaling use the same Q=128 as nnue::evaluate()
// so that training actually optimizes the function the engine runs during search.
void train_nnue(NNUEState& nnue, const std::vector<TrainingPosition>& data,
                const TrainingConfig& config, OptimizerState& optim) {
    if (data.empty()) {
        std::cout << "No training data available.\n";
        return;
    }

    std::cout << "Training on " << data.size() << " positions...\n";

    struct Sample {
        std::vector<float> features;
        float target;
    };

    std::vector<Sample> samples;
    for (const auto& pos : data) {
        Board board;
        board.set_fen(pos.fen);

        Sample s;
        pos_to_features(board, s.features);

        // evaluate() returns from side-to-move's perspective.
        // Convert to white's perspective for a consistent target.
        int current_score = evaluate(board, const_cast<NNUEState&>(nnue));
        float score_normalized = current_score / 400.0f;
        if (pos.side_to_move == 1) score_normalized = -score_normalized;

        // Target: game result (primary) blended with current engine evaluation
        s.target = (pos.result > 0) ? 1.0f : (pos.result < 0) ? 0.0f : 0.5f;
        float engine_pred = sigmoid(score_normalized);
        s.target = 0.7f * s.target + 0.3f * engine_pred;

        samples.push_back(s);
    }

    std::shuffle(samples.begin(), samples.end(), train_rng);

    size_t num_params = NNUE_INPUT * NNUE_HIDDEN1 + NNUE_HIDDEN1 +
                       NNUE_HIDDEN1 * NNUE_HIDDEN2 + NNUE_HIDDEN2 +
                       NNUE_HIDDEN2 + 1;
    if (optim.m.empty()) {
        optim.init(num_params);
    }

    optim.timestep++;
    // Learning rate: must be large enough that lr * Q > 0.5 so that weight
    // updates survive int16 quantization. With Q=128, minimum lr is ~0.004.
    float lr = config.learning_rate;
    float Qf = (float)TRAIN_Q;

    float total_loss = 0;
    int count = 0;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        float epoch_loss = 0;

        for (size_t i = 0; i < samples.size(); i += config.batch_size) {
            size_t end = std::min(i + (size_t)config.batch_size, samples.size());

            for (size_t j = i; j < end; j++) {
                const auto& sample = samples[j];

                // --- Forward pass (float, matches nnue.cpp evaluate exactly) ---

                // Hidden1: acc1 = input_bias/Q + Σ feature*input_weight/Q, ReLU
                std::array<float, NNUE_HIDDEN1> h1;
                for (int k = 0; k < NNUE_HIDDEN1; k++) {
                    float sum = (float)nnue.weights.input_bias[k] / Qf;
                    for (int f = 0; f < NNUE_INPUT; f++) {
                        if (sample.features[f] > 0)
                            sum += (float)nnue.weights.input_weights[f][k] / Qf;
                    }
                    h1[k] = sum > 0 ? sum : 0;
                }

                // Hidden2: acc2 = hidden1_bias/Q + Σ h1*hidden1_weight/Q, ReLU
                std::array<float, NNUE_HIDDEN2> h2;
                for (int k = 0; k < NNUE_HIDDEN2; k++) {
                    float sum = (float)nnue.weights.hidden1_bias[k] / Qf;
                    for (int h = 0; h < NNUE_HIDDEN1; h++) {
                        sum += h1[h] * ((float)nnue.weights.hidden1_weights[h][k] / Qf);
                    }
                    h2[k] = sum > 0 ? sum : 0;
                }

                // Output: output = output_bias/Q + Σ h2*output_weight/Q
                float output = (float)nnue.weights.output_bias / Qf;
                for (int k = 0; k < NNUE_HIDDEN2; k++) {
                    output += h2[k] * ((float)nnue.weights.output_weights[k] / Qf);
                }

                // Sigmoid with same scaling as nnue.cpp: cp = output * 200, pred = sigmoid(cp/400)
                float pred = sigmoid(output * 200.0f / 400.0f);

                // --- Loss and gradient ---
                float error = pred - sample.target;
                float loss = -sample.target * std::log(pred + 1e-7f) -
                             (1 - sample.target) * std::log(1 - pred + 1e-7f);
                epoch_loss += loss;

                // dL/d(output) = T * (pred - target) where T = 200/400 = 0.5
                float d_out = 0.5f * error;

                // --- Backward pass (all gradients in real-valued / dequantized space) ---

                // Output layer: output = bias3/Q + Σ h2[k]*w3[k]/Q
                // dL/dw3[k] = d_out * h2[k] / Q,  dL/dbias3 = d_out / Q
                for (int k = 0; k < NNUE_HIDDEN2; k++) {
                    float grad = d_out * h2[k] / Qf;
                    int idx = NNUE_INPUT * NNUE_HIDDEN1 + NNUE_HIDDEN1 +
                             NNUE_HIDDEN1 * NNUE_HIDDEN2 + NNUE_HIDDEN2 + k;
                    optim.m[idx] = 0.9f * optim.m[idx] + 0.1f * grad;
                    optim.v[idx] = 0.999f * optim.v[idx] + 0.001f * grad * grad;
                    float mh = optim.m[idx] / (1 - std::pow(0.9f, optim.timestep));
                    float vh = optim.v[idx] / (1 - std::pow(0.999f, optim.timestep));
                    float w = (float)nnue.weights.output_weights[k] / Qf;
                    w -= lr * mh / (std::sqrt(vh) + 1e-8f);
                    nnue.weights.output_weights[k] = (int16_t)std::round(w * Qf);
                }

                // Output bias
                {
                    float grad = d_out / Qf;
                    int idx = NNUE_INPUT * NNUE_HIDDEN1 + NNUE_HIDDEN1 +
                             NNUE_HIDDEN1 * NNUE_HIDDEN2 + NNUE_HIDDEN2;
                    optim.m[idx] = 0.9f * optim.m[idx] + 0.1f * grad;
                    optim.v[idx] = 0.999f * optim.v[idx] + 0.001f * grad * grad;
                    float mh = optim.m[idx] / (1 - std::pow(0.9f, optim.timestep));
                    float vh = optim.v[idx] / (1 - std::pow(0.999f, optim.timestep));
                    float w = (float)nnue.weights.output_bias / Qf;
                    w -= lr * mh / (std::sqrt(vh) + 1e-8f);
                    nnue.weights.output_bias = (int16_t)std::round(w * Qf);
                }

                // Hidden2 pre-activation gradient: dL/d_acc2[k] = d_out * w3[k]/Q * relu'(acc2[k])
                std::array<float, NNUE_HIDDEN2> d_acc2;
                for (int k = 0; k < NNUE_HIDDEN2; k++) {
                    d_acc2[k] = (h2[k] > 0) ? d_out * (float)nnue.weights.output_weights[k] / Qf : 0;
                }

                // Hidden1->Hidden2 weights: acc2[k] = bias2[k]/Q + Σ h1[h]*w2[h][k]/Q
                for (int h = 0; h < NNUE_HIDDEN1; h++) {
                    if (h1[h] <= 0) continue;
                    for (int k = 0; k < NNUE_HIDDEN2; k++) {
                        float grad = d_acc2[k] * h1[h] / Qf;
                        int idx = NNUE_INPUT * NNUE_HIDDEN1 + NNUE_HIDDEN1 +
                                 h * NNUE_HIDDEN2 + k;
                        optim.m[idx] = 0.9f * optim.m[idx] + 0.1f * grad;
                        optim.v[idx] = 0.999f * optim.v[idx] + 0.001f * grad * grad;
                        float mh = optim.m[idx] / (1 - std::pow(0.9f, optim.timestep));
                        float vh = optim.v[idx] / (1 - std::pow(0.999f, optim.timestep));
                        float w = (float)nnue.weights.hidden1_weights[h][k] / Qf;
                        w -= lr * mh / (std::sqrt(vh) + 1e-8f);
                        nnue.weights.hidden1_weights[h][k] = (int16_t)std::round(w * Qf);
                    }
                }

                // Hidden1 bias
                for (int k = 0; k < NNUE_HIDDEN2; k++) {
                    if (h2[k] <= 0) continue;
                    float grad = d_acc2[k] / Qf;
                    int idx = NNUE_INPUT * NNUE_HIDDEN1 + NNUE_HIDDEN1 +
                             NNUE_HIDDEN1 * NNUE_HIDDEN2 + k;
                    optim.m[idx] = 0.9f * optim.m[idx] + 0.1f * grad;
                    optim.v[idx] = 0.999f * optim.v[idx] + 0.001f * grad * grad;
                    float mh = optim.m[idx] / (1 - std::pow(0.9f, optim.timestep));
                    float vh = optim.v[idx] / (1 - std::pow(0.999f, optim.timestep));
                    float w = (float)nnue.weights.hidden1_bias[k] / Qf;
                    w -= lr * mh / (std::sqrt(vh) + 1e-8f);
                    nnue.weights.hidden1_bias[k] = (int16_t)std::round(w * Qf);
                }

                // Hidden1 pre-activation gradient: dL/d_acc1[h] = Σ_k d_acc2[k]*w2[h][k]/Q * relu'(acc1[h])
                std::array<float, NNUE_HIDDEN1> d_acc1;
                for (int h = 0; h < NNUE_HIDDEN1; h++) {
                    d_acc1[h] = 0;
                    if (h1[h] <= 0) continue;
                    for (int k = 0; k < NNUE_HIDDEN2; k++) {
                        d_acc1[h] += d_acc2[k] * (float)nnue.weights.hidden1_weights[h][k] / Qf;
                    }
                }

                // Input->Hidden1 weights: acc1[h] = bias1[h]/Q + Σ feature[f]*w1[f][h]/Q
                for (int f = 0; f < NNUE_INPUT; f++) {
                    if (sample.features[f] <= 0) continue;
                    for (int h = 0; h < NNUE_HIDDEN1; h++) {
                        if (h1[h] <= 0) continue;
                        float grad = d_acc1[h] / Qf;
                        int idx = f * NNUE_HIDDEN1 + h;
                        optim.m[idx] = 0.9f * optim.m[idx] + 0.1f * grad;
                        optim.v[idx] = 0.999f * optim.v[idx] + 0.001f * grad * grad;
                        float mh = optim.m[idx] / (1 - std::pow(0.9f, optim.timestep));
                        float vh = optim.v[idx] / (1 - std::pow(0.999f, optim.timestep));
                        float w = (float)nnue.weights.input_weights[f][h] / Qf;
                        w -= lr * mh / (std::sqrt(vh) + 1e-8f);
                        nnue.weights.input_weights[f][h] = (int16_t)std::round(w * Qf);
                    }
                }

                // Input bias
                for (int h = 0; h < NNUE_HIDDEN1; h++) {
                    if (h1[h] <= 0) continue;
                    float grad = d_acc1[h] / Qf;
                    int idx = NNUE_INPUT * NNUE_HIDDEN1 + h;
                    optim.m[idx] = 0.9f * optim.m[idx] + 0.1f * grad;
                    optim.v[idx] = 0.999f * optim.v[idx] + 0.001f * grad * grad;
                    float mh = optim.m[idx] / (1 - std::pow(0.9f, optim.timestep));
                    float vh = optim.v[idx] / (1 - std::pow(0.999f, optim.timestep));
                    float w = (float)nnue.weights.input_bias[h] / Qf;
                    w -= lr * mh / (std::sqrt(vh) + 1e-8f);
                    nnue.weights.input_bias[h] = (int16_t)std::round(w * Qf);
                }

                count++;
            }

            total_loss += epoch_loss;
        }

        float avg_loss = epoch_loss / std::max(1u, (unsigned)(samples.size()));
        std::cout << "  Epoch " << epoch + 1 << "/" << config.epochs
                  << " Loss: " << avg_loss << "\n";
    }

    float avg = total_loss / std::max(1, count);
    std::cout << "Training complete. Average loss: " << avg << "\n";
}

float evaluate_accuracy(const NNUEState& nnue, const std::vector<TrainingPosition>& data) {
    if (data.empty()) return 0;

    int correct = 0;

    for (const auto& pos : data) {
        Board board;
        board.set_fen(pos.fen);

        // evaluate() returns from side-to-move's perspective; convert to
        // white's perspective so it matches pos.result (white-positive).
        int score = evaluate(board, const_cast<NNUEState&>(nnue));
        float score_norm = score / 400.0f;
        if (pos.side_to_move == 1) score_norm = -score_norm;
        float pred = sigmoid(score_norm);

        int actual = pos.result;
        if ((pred > 0.5f && actual > 0) || (pred < 0.5f && actual < 0) ||
            (std::abs(pred - 0.5f) < 0.1f && actual == 0))
            correct++;
    }

    return (float)correct / data.size();
}

} // namespace neko

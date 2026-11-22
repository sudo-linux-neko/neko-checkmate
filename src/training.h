// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include "types.h"
#include "nnue.h"
#include "selfplay.h"
#include <vector>

namespace neko {

struct TrainingConfig {
    int epochs;            // training epochs per iteration
    float learning_rate;   // learning rate
    int batch_size;        // mini-batch size
    float weight_decay;    // L2 regularization
};

struct OptimizerState {
    // Adam optimizer state for each weight
    std::vector<float> m; // first moment
    std::vector<float> v; // second moment
    int timestep;

    void init(size_t num_params);
    bool load(const std::string& filename);
    void save(const std::string& filename) const;
};

// Train NNUE weights on collected self-play data
void train_nnue(NNUEState& nnue, const std::vector<TrainingPosition>& data,
                const TrainingConfig& config, OptimizerState& optim);

// Evaluate network accuracy on held-out data
float evaluate_accuracy(const NNUEState& nnue, const std::vector<TrainingPosition>& data);

// Convert position to NNUE features
void pos_to_features(const Board& board, std::vector<float>& features);

} // namespace neko

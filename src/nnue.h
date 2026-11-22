// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include "types.h"
#include "board.h"
#include <vector>
#include <array>
#include <fstream>
#include <cstdint>

namespace neko {

// Small NNUE network for evaluation
// Input: 768 features (64 squares * 6 piece types * 2 colors)
// Hidden1: 128 neurons
// Hidden2: 64 neurons
// Output: 1 (win probability)

constexpr int NNUE_INPUT = 768;
constexpr int NNUE_HIDDEN1 = 128;
constexpr int NNUE_HIDDEN2 = 64;

// Incremental accumulator for fast updates
struct Accumulator {
    std::array<int16_t, NNUE_HIDDEN1> values;
};

struct NNUEWeights {
    // Input -> Hidden1
    std::array<std::array<int16_t, NNUE_HIDDEN1>, NNUE_INPUT> input_weights;
    std::array<int16_t, NNUE_HIDDEN1> input_bias;

    // Hidden1 -> Hidden2
    std::array<std::array<int16_t, NNUE_HIDDEN2>, NNUE_HIDDEN1> hidden1_weights;
    std::array<int16_t, NNUE_HIDDEN2> hidden1_bias;

    // Hidden2 -> Output
    std::array<int16_t, NNUE_HIDDEN2> output_weights;
    int16_t output_bias;

    void randomize();
    void save(const std::string& filename) const;
    bool load(const std::string& filename);
};

struct NNUEState {
    NNUEWeights weights;
    std::vector<Accumulator> accumulator_stack;

    // Current accumulator (perspective of side to move)
    Accumulator current;

    void init();
    void reset(const Board& board);
    void push();
    void pop();

    // Update after a move
    void update_accumulator(const Board& board, Move m);

    // Evaluate the position
    int evaluate(const Board& board);
};

// Feature index: (piece_type * 2 + color) * 64 + square
inline int feature_index(Piece p, int sq) {
    return (type_of(p) * 2 + color_of(p)) * 64 + sq;
}

// Initialize NNUE
void init_nnue();

} // namespace neko

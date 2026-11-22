// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include "nnue.h"
#include <random>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>

namespace neko {

static std::mt19937 rng(42);

void NNUEWeights::randomize() {
    // Xavier-like initialization. Weights are int16 stored in quantized space.
    // With Q=128 and ~32 active features, accumulator std dev ≈ sqrt(32) * sigma_w.
    // We want accumulator values in range [-500, +500] (before /Q) so hidden1
    // outputs are in [-4, +4] range — large enough to survive two more layers.
    // sigma_w = 4.0 / sqrt(32) ≈ 0.7, so use sigma ≈ 1.0.
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < NNUE_INPUT; i++)
        for (int j = 0; j < NNUE_HIDDEN1; j++)
            input_weights[i][j] = (int16_t)std::round(dist(rng) * 4);

    for (int i = 0; i < NNUE_HIDDEN1; i++)
        input_bias[i] = (int16_t)std::round(dist(rng) * 4);

    for (int i = 0; i < NNUE_HIDDEN1; i++)
        for (int j = 0; j < NNUE_HIDDEN2; j++)
            hidden1_weights[i][j] = (int16_t)std::round(dist(rng) * 4);

    for (int i = 0; i < NNUE_HIDDEN2; i++)
        hidden1_bias[i] = (int16_t)std::round(dist(rng) * 4);

    for (int i = 0; i < NNUE_HIDDEN2; i++)
        output_weights[i] = (int16_t)std::round(dist(rng) * 4);

    output_bias = (int16_t)std::round(dist(rng) * 8);
}

void NNUEWeights::save(const std::string& filename) const {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to save NNUE weights to " << filename << "\n";
        return;
    }
    f.write(reinterpret_cast<const char*>(input_weights.data()), sizeof(input_weights));
    f.write(reinterpret_cast<const char*>(input_bias.data()), sizeof(input_bias));
    f.write(reinterpret_cast<const char*>(hidden1_weights.data()), sizeof(hidden1_weights));
    f.write(reinterpret_cast<const char*>(hidden1_bias.data()), sizeof(hidden1_bias));
    f.write(reinterpret_cast<const char*>(output_weights.data()), sizeof(output_weights));
    f.write(reinterpret_cast<const char*>(&output_bias), sizeof(output_bias));
}

bool NNUEWeights::load(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) return false;
    f.read(reinterpret_cast<char*>(input_weights.data()), sizeof(input_weights));
    f.read(reinterpret_cast<char*>(input_bias.data()), sizeof(input_bias));
    f.read(reinterpret_cast<char*>(hidden1_weights.data()), sizeof(hidden1_weights));
    f.read(reinterpret_cast<char*>(hidden1_bias.data()), sizeof(hidden1_bias));
    f.read(reinterpret_cast<char*>(output_weights.data()), sizeof(output_weights));
    f.read(reinterpret_cast<char*>(&output_bias), sizeof(output_bias));
    return f.good();
}

// Bug 4 fix: single shared quantization scale Q = 128.
// Accumulator holds int16 values in "quantized space" where each unit = 1/Q.
// To dequantize: float_val = int_val / Q.
// To quantize:   int_val = float_val * Q.
// We multiply features (0 or 1) by weights (small ints), accumulate, then
// dequantize once at the end with /Q. No double-division.
static constexpr int Q = 128;

void NNUEState::init() {
    weights.randomize();
    accumulator_stack.clear();
    current.values.fill(0);
}

void NNUEState::reset(const Board& board) {
    // Bug 4 fix: accumulator = input_bias + sum over active features of input_weights[feat][j]
    // This is the standard NNUE accumulator: dot(feature_vector, input_column_j) + bias_j
    current.values = weights.input_bias;
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board.piece_on(sq);
        if (p != NO_PIECE) {
            int idx = feature_index(p, sq);
            for (int j = 0; j < NNUE_HIDDEN1; j++)
                current.values[j] += weights.input_weights[idx][j];
        }
    }
    accumulator_stack.clear();
    accumulator_stack.push_back(current);
}

void NNUEState::push() {
    accumulator_stack.push_back(current);
}

void NNUEState::pop() {
    if (!accumulator_stack.empty()) {
        current = accumulator_stack.back();
        accumulator_stack.pop_back();
    }
}

// True incremental update — diff piece changes from the move and adjust
// the accumulator by subtracting removed features and adding new ones.
// This is O(changed_pieces * HIDDEN1) instead of O(64 * HIDDEN1) for a full reset.
void NNUEState::update_accumulator(const Board& board, Move m) {
    Color us = flip(board.side); // the side that just moved (board.side already flipped)

    // Remove piece from 'from' square
    Piece moving = Piece(m.piece);
    int feat_from = feature_index(moving, m.from);
    for (int j = 0; j < NNUE_HIDDEN1; j++)
        current.values[j] -= weights.input_weights[feat_from][j];

    // Add piece on 'to' square (may be promoted)
    PieceType promo_type = (m.promotion != NO_PIECE) ? type_of(Piece(m.promotion)) : type_of(moving);
    Piece placed = make_piece(us, promo_type);
    int feat_to = feature_index(placed, m.to);
    for (int j = 0; j < NNUE_HIDDEN1; j++)
        current.values[j] += weights.input_weights[feat_to][j];

    // Normal capture: remove captured piece from 'to' (not for EP — handled below)
    if (m.captured != NO_PIECE && m.move_type != 2) {
        int feat_cap = feature_index(Piece(m.captured), m.to);
        for (int j = 0; j < NNUE_HIDDEN1; j++)
            current.values[j] -= weights.input_weights[feat_cap][j];
    }

    // En passant capture: remove the captured pawn from its actual square
    if (m.move_type == 2) {
        int captured_sq = (us == WHITE) ? m.to - 8 : m.to + 8;
        Piece ep_pawn = make_piece(flip(us), PAWN);
        int feat_ep = feature_index(ep_pawn, captured_sq);
        for (int j = 0; j < NNUE_HIDDEN1; j++)
            current.values[j] -= weights.input_weights[feat_ep][j];
    }

    // Castling: also move the rook
    if (m.move_type == 1) {
        int rook_from = -1, rook_to = -1;
        if (m.to == G1) { rook_from = H1; rook_to = F1; }
        else if (m.to == C1) { rook_from = A1; rook_to = D1; }
        else if (m.to == G8) { rook_from = H8; rook_to = F8; }
        else if (m.to == C8) { rook_from = A8; rook_to = D8; }

        if (rook_from >= 0) {
            Piece rook = make_piece(us, ROOK);
            int feat_rf = feature_index(rook, rook_from);
            int feat_rt = feature_index(rook, rook_to);
            for (int j = 0; j < NNUE_HIDDEN1; j++) {
                current.values[j] -= weights.input_weights[feat_rf][j];
                current.values[j] += weights.input_weights[feat_rt][j];
            }
        }
    }
}

int NNUEState::evaluate(const Board& board) {
    // The accumulator (current.values) already holds:
    //   acc[j] = input_bias[j] + sum over active features of input_weights[feat][j]
    // in quantized int16 space (scale Q=128).

    // Hidden1: dequantize accumulator, apply ReLU
    std::array<float, NNUE_HIDDEN1> h1;
    for (int j = 0; j < NNUE_HIDDEN1; j++) {
        float val = (float)current.values[j] / (float)Q;
        h1[j] = val > 0 ? val : 0;
    }

    // Hidden2: h1 dot hidden1_weights (dequantized), bias, ReLU
    std::array<float, NNUE_HIDDEN2> h2;
    for (int j = 0; j < NNUE_HIDDEN2; j++) {
        float sum = (float)weights.hidden1_bias[j] / (float)Q;
        for (int i = 0; i < NNUE_HIDDEN1; i++) {
            sum += h1[i] * ((float)weights.hidden1_weights[i][j] / (float)Q);
        }
        h2[j] = sum > 0 ? sum : 0;
    }

    // Output: h2 dot output_weights (dequantized), bias
    float output = (float)weights.output_bias / (float)Q;
    for (int i = 0; i < NNUE_HIDDEN2; i++) {
        output += h2[i] * ((float)weights.output_weights[i] / (float)Q);
    }

    // Scale to centipawns with clamping
    float cp = output * 200.0f;
    if (cp > 3000.0f) cp = 3000.0f;
    if (cp < -3000.0f) cp = -3000.0f;
    return (int)std::round(cp);
}

void init_nnue() {
}

} // namespace neko

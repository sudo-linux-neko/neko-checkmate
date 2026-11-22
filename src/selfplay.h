// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include "types.h"
#include "board.h"
#include "nnue.h"
#include "search.h"
#include <vector>

namespace neko {

// A single position from self-play with its game result
struct TrainingPosition {
    std::string fen;
    int score;   // engine score in centipawns
    int result;  // +1 white wins, 0 draw, -1 black wins
    int side_to_move; // 0=white, 1=black
};

// A complete self-play game
struct SelfPlayGame {
    std::vector<TrainingPosition> positions;
    int result; // +1 white wins, 0 draw, -1 black wins
};

struct SelfPlayConfig {
    int num_games;          // games to play per iteration
    int search_depth;       // search depth for self-play
    int time_per_move_ms;   // time limit per move
    int max_moves_per_game; // max moves before draw
    int num_iterations;     // total training iterations
    int save_every;         // save checkpoint every N iterations
};

struct SelfPlayState {
    SelfPlayConfig config;
    int current_iteration;
    int total_games_played;
    int total_positions;
    std::vector<TrainingPosition> training_buffer;

    void init(const SelfPlayConfig& cfg);
    bool load_checkpoint(const std::string& directory);
    void save_checkpoint(const std::string& directory) const;
};

// Play a single self-play game
SelfPlayGame play_game(NNUEState& nnue, const SelfPlayConfig& config, SearchInfo& info);

// Run the self-play training loop
void run_self_play(const SelfPlayConfig& config);

// Save training data to file
void save_training_data(const std::vector<TrainingPosition>& data, const std::string& filename);

// Load training data from file
std::vector<TrainingPosition> load_training_data(const std::string& filename);

} // namespace neko

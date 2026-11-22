// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include "selfplay.h"
#include "evaluate.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <filesystem>
#include <chrono>
#include <csignal>

namespace neko {

static std::mt19937 selfplay_rng(12345);

void SelfPlayState::init(const SelfPlayConfig& cfg) {
    config = cfg;
    current_iteration = 0;
    total_games_played = 0;
    total_positions = 0;
    training_buffer.clear();
}

bool SelfPlayState::load_checkpoint(const std::string& dir) {
    std::string state_file = dir + "/selfplay_state.bin";
    std::ifstream f(state_file, std::ios::binary);
    if (!f.is_open()) return false;

    f.read(reinterpret_cast<char*>(&current_iteration), sizeof(current_iteration));
    f.read(reinterpret_cast<char*>(&total_games_played), sizeof(total_games_played));
    f.read(reinterpret_cast<char*>(&total_positions), sizeof(total_positions));

    // Load training buffer size and data
    int buf_size = 0;
    f.read(reinterpret_cast<char*>(&buf_size), sizeof(buf_size));
    training_buffer.resize(buf_size);
    for (int i = 0; i < buf_size; i++) {
        auto& pos = training_buffer[i];
        // Read fen length
        int fen_len = 0;
        f.read(reinterpret_cast<char*>(&fen_len), sizeof(fen_len));
        pos.fen.resize(fen_len);
        f.read(&pos.fen[0], fen_len);
        f.read(reinterpret_cast<char*>(&pos.score), sizeof(pos.score));
        f.read(reinterpret_cast<char*>(&pos.result), sizeof(pos.result));
        f.read(reinterpret_cast<char*>(&pos.side_to_move), sizeof(pos.side_to_move));
    }

    std::cout << "Loaded self-play checkpoint: iter=" << current_iteration
              << " games=" << total_games_played
              << " positions=" << total_positions << "\n";
    return true;
}

void SelfPlayState::save_checkpoint(const std::string& dir) const {
    std::filesystem::create_directories(dir);

    std::string state_file = dir + "/selfplay_state.bin";
    std::ofstream f(state_file, std::ios::binary);

    f.write(reinterpret_cast<const char*>(&current_iteration), sizeof(current_iteration));
    f.write(reinterpret_cast<const char*>(&total_games_played), sizeof(total_games_played));
    f.write(reinterpret_cast<const char*>(&total_positions), sizeof(total_positions));

    int buf_size = training_buffer.size();
    f.write(reinterpret_cast<const char*>(&buf_size), sizeof(buf_size));
    for (const auto& pos : training_buffer) {
        int fen_len = pos.fen.size();
        f.write(reinterpret_cast<const char*>(&fen_len), sizeof(fen_len));
        f.write(pos.fen.c_str(), fen_len);
        f.write(reinterpret_cast<const char*>(&pos.score), sizeof(pos.score));
        f.write(reinterpret_cast<const char*>(&pos.result), sizeof(pos.result));
        f.write(reinterpret_cast<const char*>(&pos.side_to_move), sizeof(pos.side_to_move));
    }

    std::cout << "Saved self-play checkpoint: iter=" << current_iteration
              << " games=" << total_games_played
              << " positions=" << total_positions << "\n";
}

SelfPlayGame play_game(NNUEState& nnue, const SelfPlayConfig& config, SearchInfo& info) {
    SelfPlayGame game;
    game.result = 0;

    Board board;
    board.set_startpos();
    nnue.reset(board);

    std::uniform_real_distribution<float> perturb_dist(-10.0f, 10.0f);

    for (int move_num = 0; move_num < config.max_moves_per_game; move_num++) {
        // Record position
        TrainingPosition pos;
        pos.fen = board.get_fen();
        pos.side_to_move = board.side;

        // Search
        info.clear();
        info.max_depth = config.search_depth;
        info.time_limit_ms = config.time_per_move_ms;

        SearchResult result = iterative_deepening(board, nnue, info);
        pos.score = result.score;

        if (result.best_move.is_null()) {
            // No legal moves
            if (board.in_check()) {
                game.result = (board.side == WHITE) ? -1 : 1;
            } else {
                game.result = 0; // stalemate
            }
            // Bug 10 fix: apply perturbation consistently even on terminal positions
            pos.score += (int)perturb_dist(selfplay_rng);
            pos.result = game.result;
            game.positions.push_back(pos);
            break;
        }

        // Make the move
        MoveList ml;
        generate_moves(board, ml);
        Move chosen = result.best_move;
        bool found = false;
        for (int i = 0; i < ml.count; i++) {
            if (ml.moves[i] == chosen) {
                chosen = ml.moves[i];
                found = true;
                break;
            }
        }
        if (!found && ml.count > 0) {
            chosen = ml.moves[0];
        }

        board.make_move(chosen);
        nnue.update_accumulator(board, chosen);

        // Check for draw by 50-move rule
        if (board.halfmove >= 100) {
            game.result = 0;
            pos.result = 0;
            game.positions.push_back(pos);
            break;
        }

        // Add small random perturbation to evaluation for variety
        pos.score += (int)perturb_dist(selfplay_rng);
        game.positions.push_back(pos);
    }

    // If game didn't end, declare draw
    if (game.result == 0 && !game.positions.empty()) {
        game.positions.back().result = 0;
    }

    // Set result for all positions
    for (auto& pos : game.positions) {
        pos.result = game.result;
    }

    return game;
}

void save_training_data(const std::vector<TrainingPosition>& data, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Failed to save training data to " << filename << "\n";
        return;
    }
    for (const auto& pos : data) {
        f << pos.fen << "|" << pos.score << "|" << pos.result << "|" << pos.side_to_move << "\n";
    }
    std::cout << "Saved " << data.size() << " positions to " << filename << "\n";
}

std::vector<TrainingPosition> load_training_data(const std::string& filename) {
    std::vector<TrainingPosition> data;
    std::ifstream f(filename);
    if (!f.is_open()) return data;

    std::string line;
    while (std::getline(f, line)) {
        TrainingPosition pos;
        std::istringstream ss(line);
        std::string token;
        std::getline(ss, pos.fen, '|');
        std::getline(ss, token, '|'); pos.score = std::stoi(token);
        std::getline(ss, token, '|'); pos.result = std::stoi(token);
        std::getline(ss, token, '|'); pos.side_to_move = std::stoi(token);
        data.push_back(pos);
    }
    return data;
}

// Simple signal handling for safe interruption
static volatile bool g_stop_requested = false;

void signal_handler(int sig) {
    g_stop_requested = true;
    std::cout << "\nStop requested. Saving checkpoint...\n";
}

void run_self_play(const SelfPlayConfig& config) {
    // Install signal handler
    signal(SIGINT, signal_handler);

    std::string checkpoint_dir = "checkpoint";
    std::string nnue_file = checkpoint_dir + "/nnue.bin";
    std::string train_file = checkpoint_dir + "/training_data.txt";

    SelfPlayState state;
    state.init(config);

    NNUEState nnue;
    nnue.init();

    // Try to load checkpoint
    if (state.load_checkpoint(checkpoint_dir)) {
        if (std::filesystem::exists(nnue_file)) {
            nnue.weights.load(nnue_file);
            std::cout << "Loaded NNUE weights from checkpoint\n";
        }
    }

    // Load existing training data
    auto existing_data = load_training_data(train_file);
    state.training_buffer.insert(state.training_buffer.end(),
                                  existing_data.begin(), existing_data.end());

    std::cout << "Starting self-play training from iteration " << state.current_iteration << "\n";
    std::cout << "Config: " << config.num_iterations << " iterations, "
              << config.num_games << " games/iter, depth " << config.search_depth << "\n";

    SearchInfo info;

    for (int iter = state.current_iteration; iter < config.num_iterations; iter++) {
        if (g_stop_requested) break;

        state.current_iteration = iter + 1;
        std::cout << "\n=== Iteration " << state.current_iteration << " ===\n";

        int games_this_iter = 0;
        for (int g = 0; g < config.num_games; g++) {
            if (g_stop_requested) break;

            SelfPlayGame game = play_game(nnue, config, info);

            for (const auto& pos : game.positions) {
                state.training_buffer.push_back(pos);
            }

            state.total_games_played++;
            games_this_iter++;
            state.total_positions += game.positions.size();

            if ((g + 1) % 10 == 0) {
                std::cout << "  Games: " << games_this_iter << "/" << config.num_games
                          << " Total positions: " << state.total_positions << "\n";
            }
        }

        std::cout << "Iteration " << state.current_iteration << " complete. "
                  << "Games: " << games_this_iter
                  << " Total: " << state.total_games_played << "\n";

        // Save checkpoint periodically
        if (state.current_iteration % config.save_every == 0 || g_stop_requested) {
            state.save_checkpoint(checkpoint_dir);
            nnue.weights.save(nnue_file);
            save_training_data(state.training_buffer, train_file);
            std::cout << "Checkpoint saved.\n";
        }
    }

    // Final save
    state.save_checkpoint(checkpoint_dir);
    nnue.weights.save(nnue_file);
    save_training_data(state.training_buffer, train_file);
    std::cout << "Training complete.\n";
}

} // namespace neko

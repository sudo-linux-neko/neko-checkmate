// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <filesystem>
#include "types.h"
#include "bitboard.h"
#include "board.h"
#include "nnue.h"
#include "search.h"
#include "selfplay.h"
#include "training.h"
#include "uci.h"
#include "evaluate.h"

void print_usage() {
    std::cout << "NekoMeow Chess Engine v1.0\n\n";
    std::cout << "Usage:\n";
    std::cout << "  neko_meow uci          - Start UCI protocol mode\n";
    std::cout << "  neko_meow selfplay     - Start self-play training\n";
    std::cout << "  neko_meow train        - Train NNUE on existing data\n";
    std::cout << "  neko_meow bench        - Run benchmark\n";
    std::cout << "\nSelf-play options:\n";
    std::cout << "  --games N              - Games per iteration (default: 20)\n";
    std::cout << "  --depth N              - Search depth (default: 6)\n";
    std::cout << "  --iterations N         - Total iterations (default: 100)\n";
    std::cout << "  --save-every N         - Save checkpoint every N iterations (default: 5)\n";
}

void run_benchmark() {
    neko::Board board;
    board.set_startpos();
    board.print();

    neko::NNUEState nnue;
    nnue.init();
    // Bug 3 fix: load trained weights if checkpoint exists
    if (std::filesystem::exists("checkpoint/nnue.bin")) {
        nnue.weights.load("checkpoint/nnue.bin");
        std::cout << "Loaded NNUE weights from checkpoint.\n";
    }
    nnue.reset(board);

    neko::SearchInfo info;
    info.max_depth = 12;
    info.time_limit_ms = 5000;

    std::cout << "Running benchmark...\n";
    auto start = std::chrono::steady_clock::now();

    neko::SearchResult result = neko::iterative_deepening(board, nnue, info);

    auto end = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(end - start).count();

    std::cout << "\nBenchmark results:\n";
    std::cout << "Depth: " << result.depth << "\n";
    std::cout << "Nodes: " << result.nodes << "\n";
    std::cout << "Time: " << elapsed << "s\n";
    std::cout << "NPS: " << (int)(result.nodes / elapsed) << "\n";
    std::cout << "Score: " << result.score << " cp\n";
    if (!result.best_move.is_null()) {
        std::string move_str;
        move_str += char(neko::file_of(result.best_move.from) + 'a');
        move_str += char(neko::rank_of(result.best_move.from) + '1');
        move_str += char(neko::file_of(result.best_move.to) + 'a');
        move_str += char(neko::rank_of(result.best_move.to) + '1');
        std::cout << "Best move: " << move_str << "\n";
    }
}

int main(int argc, char* argv[]) {
    neko::init_bitboards();
    neko::init_zobrist();
    neko::init_nnue();
    neko::TT.init();

    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];

    if (command == "uci") {
        neko::UCI uci;
        uci.run();
    }
    else if (command == "selfplay") {
        neko::SelfPlayConfig config;
        config.num_games = 20;
        config.search_depth = 6;
        config.time_per_move_ms = 500;
        config.max_moves_per_game = 200;
        config.num_iterations = 100;
        config.save_every = 5;

        // Parse arguments
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--games" && i + 1 < argc) config.num_games = std::stoi(argv[++i]);
            else if (arg == "--depth" && i + 1 < argc) config.search_depth = std::stoi(argv[++i]);
            else if (arg == "--iterations" && i + 1 < argc) config.num_iterations = std::stoi(argv[++i]);
            else if (arg == "--save-every" && i + 1 < argc) config.save_every = std::stoi(argv[++i]);
            else if (arg == "--time" && i + 1 < argc) config.time_per_move_ms = std::stoi(argv[++i]);
        }

        neko::run_self_play(config);
    }
    else if (command == "train") {
        neko::SelfPlayConfig config;
        config.num_games = 0;
        config.search_depth = 0;
        config.time_per_move_ms = 0;
        config.max_moves_per_game = 0;
        config.num_iterations = 0;
        config.save_every = 1;

        std::string checkpoint_dir = "checkpoint";
        std::string train_file = checkpoint_dir + "/training_data.txt";
        std::string nnue_file = checkpoint_dir + "/nnue.bin";
        std::string optim_file = checkpoint_dir + "/optimizer.bin";

        // Load training data
        auto data = neko::load_training_data(train_file);
        if (data.empty()) {
            std::cout << "No training data found. Run selfplay first.\n";
            return 1;
        }

        neko::NNUEState nnue;
        nnue.init();
        if (std::filesystem::exists(nnue_file)) {
            nnue.weights.load(nnue_file);
            std::cout << "Loaded existing NNUE weights.\n";
        }

        neko::OptimizerState optim;
        optim.load(optim_file);

        neko::TrainingConfig tconfig;
        tconfig.epochs = 10;
        tconfig.learning_rate = 0.01f;
        tconfig.batch_size = 32;
        tconfig.weight_decay = 0.0001f;

        // Parse arguments
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--epochs" && i + 1 < argc) tconfig.epochs = std::stoi(argv[++i]);
            else if (arg == "--lr" && i + 1 < argc) tconfig.learning_rate = std::stof(argv[++i]);
            else if (arg == "--batch-size" && i + 1 < argc) tconfig.batch_size = std::stoi(argv[++i]);
        }

        neko::train_nnue(nnue, data, tconfig, optim);

        nnue.weights.save(nnue_file);
        optim.save(optim_file);
        std::cout << "Model saved.\n";
    }
    else if (command == "bench") {
        run_benchmark();
    }
    else {
        print_usage();
        return 1;
    }

    return 0;
}

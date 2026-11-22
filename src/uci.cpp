// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include "uci.h"
#include "evaluate.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <atomic>

namespace neko {

static u64 now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

UCI::UCI() : pondering(false), infinite(false), num_threads(1) {
    init_bitboards();
    init_zobrist();
    init_nnue();
    TT.init();
    board.set_startpos();
    nnue.init();
    if (std::filesystem::exists("checkpoint/nnue.bin")) {
        nnue.weights.load("checkpoint/nnue.bin");
    }
    nnue.reset(board);
}

UCI::~UCI() {
    search_info.stopped = true;
    for (auto& t : worker_threads)
        if (t.joinable()) t.join();
    if (search_thread.joinable())
        search_thread.join();
}

void UCI::run() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        process_command(line);
    }
}

void UCI::process_command(const std::string& command) {
    std::istringstream ss(command);
    std::string token;
    ss >> token;

    if (token == "uci") cmd_uci();
    else if (token == "isready") cmd_isready();
    else if (token == "ucinewgame") cmd_ucinewgame();
    else if (token == "position") {
        std::string args;
        std::getline(ss, args);
        cmd_position(args);
    }
    else if (token == "go") {
        std::string args;
        std::getline(ss, args);
        cmd_go(args);
    }
    else if (token == "stop") cmd_stop();
    else if (token == "quit") cmd_quit();
    else if (token == "d") {
        board.print();
    }
    else if (token == "eval") {
        int score = evaluate(board, nnue);
        std::cout << "info string Eval: " << score << " cp\n";
    }
    else if (token == "setoption") {
        std::string name, value;
        ss >> name;
        if (name == "name") {
            std::string opt_name;
            ss >> opt_name;
            if (opt_name == "Threads") {
                ss >> value >> value;
                num_threads = std::clamp(std::stoi(value), 1, 8);
                std::cout << "info string Threads set to " << num_threads << "\n";
            } else if (opt_name == "Hash") {
                ss >> value >> value;
                int mb = std::clamp(std::stoi(value), 1, 4096);
                std::cout << "info string Hash set to " << mb << "MB\n";
            }
        }
    }
}

void UCI::cmd_uci() {
    std::cout << "id name NekoMeow v2.0\n";
    std::cout << "id author Neko\n";
    std::cout << "option name Hash type spin default 512 min 1 max 4096\n";
    std::cout << "option name Threads type spin default 1 min 1 max 8\n";
    std::cout << "uciok\n";
}

void UCI::cmd_isready() {
    std::cout << "readyok\n";
}

void UCI::cmd_ucinewgame() {
    board.set_startpos();
    nnue.reset(board);
    TT.clear();
}

void UCI::cmd_position(const std::string& args) {
    std::istringstream ss(args);
    std::string token;
    ss >> token;

    if (token == "startpos") {
        board.set_startpos();
        ss >> token;
    } else if (token == "fen") {
        std::string fen;
        while (ss >> token && token != "moves") {
            if (!fen.empty()) fen += " ";
            fen += token;
        }
        board.set_fen(fen);
    }

    if (token == "moves") {
        while (ss >> token) {
            MoveList ml;
            generate_moves(board, ml);

            int from_file = token[0] - 'a';
            int from_rank = token[1] - '1';
            int to_file = token[2] - 'a';
            int to_rank = token[3] - '1';
            int from = make_square(from_file, from_rank);
            int to = make_square(to_file, to_rank);

            int promo_type = NO_PIECE;
            if (token.size() == 5) {
                switch (token[4]) {
                    case 'q': promo_type = QUEEN; break;
                    case 'r': promo_type = ROOK; break;
                    case 'b': promo_type = BISHOP; break;
                    case 'n': promo_type = KNIGHT; break;
                }
            }

            for (int i = 0; i < ml.count; i++) {
                Move m = ml.moves[i];
                if (m.from == from && m.to == to) {
                    if (promo_type != NO_PIECE) {
                        if (m.promotion != NO_PIECE && type_of(Piece(m.promotion)) == promo_type) {
                            board.make_move(m);
                            nnue.update_accumulator(board, m);
                            break;
                        }
                    } else {
                        board.make_move(m);
                        nnue.update_accumulator(board, m);
                        break;
                    }
                }
            }
        }
    }

    nnue.reset(board);
}

void UCI::cmd_go(const std::string& args) {
    std::istringstream ss(args);
    std::string token;
    int depth = 64;
    int movetime = 1000;
    int wtime = 0, btime = 0, winc = 0, binc = 0, movestogo = 0;
    bool infinite = false;

    while (ss >> token) {
        if (token == "depth") ss >> depth;
        else if (token == "movetime") ss >> movetime;
        else if (token == "wtime") ss >> wtime;
        else if (token == "btime") ss >> btime;
        else if (token == "winc") ss >> winc;
        else if (token == "binc") ss >> binc;
        else if (token == "movestogo") ss >> movestogo;
        else if (token == "infinite") infinite = true;
    }

    int time_for_move = movetime;
    if (wtime > 0 || btime > 0) {
        int my_time = (board.side == WHITE) ? wtime : btime;
        int my_inc = (board.side == WHITE) ? winc : binc;
        if (movestogo > 0) {
            time_for_move = my_time / movestogo + my_inc / 2;
        } else {
            time_for_move = my_time / 30 + my_inc / 2;
        }
        time_for_move = std::max(time_for_move, 50);
        time_for_move = std::min(time_for_move, my_time / 2);
    }

    if (infinite) time_for_move = 3600000;

    start_search(depth, time_for_move, wtime, btime, winc, binc, movestogo);
}

void UCI::start_search(int depth, int movetime, int wtime, int btime,
                        int winc, int binc, int movestogo) {
    search_info.stopped = true;
    for (auto& t : worker_threads)
        if (t.joinable()) t.join();
    worker_threads.clear();
    if (search_thread.joinable())
        search_thread.join();

    search_info.clear();
    search_info.max_depth = depth;
    search_info.time_limit_ms = movetime;

    // Lazy SMP: spawn worker threads
    // Each thread gets its own Board + NNUEState, shares the TT
    // Workers check shared_stopped every 4096 nodes
    for (int t = 1; t < num_threads; t++) {
        Board thread_board = board;
        worker_threads.emplace_back(&UCI::worker_thread_func, this,
                                    thread_board, depth, movetime);
    }

    search_thread = std::thread(&UCI::search_thread_func, this, depth, movetime);
}

void UCI::search_thread_func(int depth, int movetime) {
    SearchResult result = iterative_deepening(board, nnue, search_info);

    search_info.stopped = true;
    for (auto& t : worker_threads)
        if (t.joinable()) t.join();
    worker_threads.clear();

    if (!result.best_move.is_null()) {
        std::string move_str;
        move_str += char(file_of(result.best_move.from) + 'a');
        move_str += char(rank_of(result.best_move.from) + '1');
        move_str += char(file_of(result.best_move.to) + 'a');
        move_str += char(rank_of(result.best_move.to) + '1');
        if (result.best_move.promotion != NO_PIECE) {
            char promo_char;
            switch (type_of(Piece(result.best_move.promotion))) {
                case QUEEN:  promo_char = 'q'; break;
                case ROOK:   promo_char = 'r'; break;
                case BISHOP: promo_char = 'b'; break;
                case KNIGHT: promo_char = 'n'; break;
                default:     promo_char = 'q'; break;
            }
            move_str += promo_char;
        }
        std::cout << "bestmove " << move_str << "\n";
    } else {
        std::cout << "bestmove 0000\n";
    }
}

void UCI::worker_thread_func(Board board, int depth, int movetime) {
    SearchInfo local_info;
    local_info.clear();
    local_info.max_depth = depth;
    local_info.time_limit_ms = movetime;
    local_info.start_time = now_ms();
    local_info.shared_stopped = &search_info.stopped;

    NNUEState local_nnue;
    local_nnue.init();
    if (std::filesystem::exists("checkpoint/nnue.bin"))
        local_nnue.weights.load("checkpoint/nnue.bin");
    local_nnue.reset(board);

    iterative_deepening(board, local_nnue, local_info);
}

void UCI::cmd_stop() {
    search_info.stopped = true;
    if (search_thread.joinable())
        search_thread.join();
}

void UCI::cmd_quit() {
    search_info.stopped = true;
    for (auto& t : worker_threads)
        if (t.joinable()) t.join();
    if (search_thread.joinable())
        search_thread.join();
    exit(0);
}

} // namespace neko

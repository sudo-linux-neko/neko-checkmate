// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include "types.h"
#include "board.h"
#include "nnue.h"
#include "search.h"
#include <string>
#include <thread>
#include <vector>

namespace neko {

class UCI {
public:
    UCI();
    ~UCI();

    void run();
    void process_command(const std::string& command);

private:
    Board board;
    NNUEState nnue;
    SearchInfo search_info;
    std::thread search_thread;
    std::vector<std::thread> worker_threads;
    int num_threads;

    bool pondering;
    bool infinite;

    void cmd_uci();
    void cmd_isready();
    void cmd_ucinewgame();
    void cmd_position(const std::string& args);
    void cmd_go(const std::string& args);
    void cmd_stop();
    void cmd_quit();

    void start_search(int depth, int movetime, int wtime, int btime,
                      int winc, int binc, int movestogo);
    void search_thread_func(int depth, int movetime);
    void worker_thread_func(Board board, int depth, int movetime);
};

} // namespace neko

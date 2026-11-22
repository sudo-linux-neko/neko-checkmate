// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include "types.h"
#include "board.h"
#include "nnue.h"
#include "movegen.h"
#include <atomic>
#include <array>
#include <vector>

namespace neko {

struct TTEntry {
    u64 key;
    int depth;
    int score;
    int flag; // 0=exact, 1=alpha cutoff, 2=beta cutoff
    Move best_move;
    int age;
};

// ~16M entries * 32 bytes = 512MB — fits in RAM, some cache misses expected
// Use 1M entries (32KB) for better cache behavior, or 4M (128KB) for L2
constexpr int TT_SIZE_BITS = 20;
constexpr int TT_SIZE = 1 << TT_SIZE_BITS;

struct TranspositionTable {
    std::vector<TTEntry> table;
    int age;

    void init();
    void clear();
    bool probe(u64 key, TTEntry& entry);
    void prefetch(u64 key) const;
    void store(u64 key, int depth, int score, int flag, Move best_move);
};

struct SearchInfo {
    std::atomic<bool> stopped;
    std::atomic<int> nodes;
    int max_depth;
    int time_limit_ms;
    u64 start_time;

    // For lazy SMP: pointer to a shared stopped flag
    std::atomic<bool>* shared_stopped = nullptr;

    std::array<std::array<Move, 2>, MAX_PLY> killers;
    std::array<std::array<int, 64>, 64> history;
    int sel_depth;

    bool should_stop() const {
        return stopped.load(std::memory_order_relaxed);
    }

    void clear();
};

struct SearchResult {
    Move best_move;
    int score;
    int depth;
    int nodes;
};

SearchResult iterative_deepening(Board& board, NNUEState& nnue, SearchInfo& info);
int alpha_beta(Board& board, NNUEState& nnue, SearchInfo& info,
               int alpha, int beta, int depth, int ply);
int quiescence(Board& board, NNUEState& nnue, SearchInfo& info,
               int alpha, int beta, int ply, int qs_ply);

void order_moves(const Board& board, MoveList& ml, const TTEntry& tt,
                 const SearchInfo& info, int ply);

int see(const Board& board, Move m);
int see_sign(const Board& board, Move m);

extern TranspositionTable TT;

} // namespace neko

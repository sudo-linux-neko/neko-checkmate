// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include "types.h"
#include "bitboard.h"
#include <array>
#include <string>
#include <vector>

namespace neko {

// Minimal state needed to undo a move (Bug 7 fix: avoid FEN round-trip)
struct UndoInfo {
    int castling;
    int ep_square;
    int halfmove;
    u64 zobrist;
    int captured_piece; // NO_PIECE if none (including ep captures)
};

struct Board {
    std::array<u64, 12> pieces;
    std::array<u64, 2>  occupancy;
    u64 all_occ;
    Color side;
    int castling;
    int ep_square;
    int halfmove;
    int fullmove;
    u64 zobrist;

    // Repetition detection: stack of Zobrist keys since last irreversible move.
    std::vector<u64> position_history;

    void set_startpos();
    void set_fen(const std::string& fen);
    std::string get_fen() const;

    void make_move(Move m);
    void undo_move(Move m, const UndoInfo& u);
    void make_null_move();
    void undo_null_move(int old_ep, u64 old_zobrist);

    // Count how many times the current position appears in history (including current)
    int repetitions() const;

    Piece piece_at(int sq) const;
    bool is_attacked(int sq, Color by) const;
    bool in_check() const;
    u64 get_piece_attacks(int sq) const;
    Piece piece_on(int sq) const;
    void print() const;
};

void init_zobrist();
u64 compute_zobrist(const Board& b);

} // namespace neko

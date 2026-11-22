// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include "types.h"
#include "board.h"

namespace neko {

struct MoveList {
    Move moves[MAX_MOVES];
    int count;
    int scores[MAX_MOVES]; // for move ordering

    void add(Move m) {
        if (count < MAX_MOVES)
            moves[count++] = m;
    }
};

// Generate all pseudo-legal moves for the side to move
void generate_moves(const Board& board, MoveList& ml);

// Generate only capture moves (for quiescence search)
void generate_captures(const Board& board, MoveList& ml);

// Generate only quiet moves
void generate_quiets(const Board& board, MoveList& ml);

} // namespace neko

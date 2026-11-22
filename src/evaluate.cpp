// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include "evaluate.h"

namespace neko {

int evaluate(const Board& board, NNUEState& nnue) {
    int score = nnue.evaluate(board);
    return (board.side == WHITE) ? score : -score;
}

int static_eval(const Board& board) {
    int score = 0;
    for (int p = 0; p < 12; p++) {
        int pt = type_of(Piece(p));
        int val = PieceValues[pt];
        int pop = popcount(board.pieces[p]);
        if (color_of(Piece(p)) == WHITE)
            score += val * pop;
        else
            score -= val * pop;
    }
    return (board.side == WHITE) ? score : -score;
}

} // namespace neko

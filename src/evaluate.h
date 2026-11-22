// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include "types.h"
#include "board.h"
#include "nnue.h"

namespace neko {

// Static piece values (in centipawns)
constexpr int PieceValues[6] = { 100, 320, 330, 500, 900, 0 };

// Simple evaluation using NNUE
int evaluate(const Board& board, NNUEState& nnue);

// Quick static eval (no NNUE, for move ordering etc)
int static_eval(const Board& board);

} // namespace neko

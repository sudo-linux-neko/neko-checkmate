// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include "types.h"
#include <array>

namespace neko {

constexpr int popcount(u64 b) { return __builtin_popcountll(b); }
constexpr int lsb(u64 b) { return __builtin_ctzll(b); }
int msb(u64 b);
int pop_lsb(u64& b);

constexpr bool more_than_one(u64 b) { return b & (b - 1); }

u64 pawn_attacks(int sq, Color c);

extern std::array<u64, 64> KnightAttacks;
extern std::array<u64, 64> KingAttacks;

void init_bitboards();

u64 bishop_attacks(int sq, u64 blockers);
u64 rook_attacks(int sq, u64 blockers);

void print_bitboard(u64 b);

} // namespace neko

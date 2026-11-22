// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include "bitboard.h"
#include <iostream>

namespace neko {

std::array<u64, 64> KnightAttacks;
std::array<u64, 64> KingAttacks;

int msb(u64 b) {
    if (b == 0) return -1;
    return 63 - __builtin_clzll(b);
}

int pop_lsb(u64& b) {
    int sq = lsb(b);
    b &= b - 1;
    return sq;
}

u64 pawn_attacks(int sq, Color c) {
    u64 b = 1ULL << sq;
    u64 left  = (c == WHITE) ? (b << 7) & ~FileHBB : (b >> 7) & ~FileABB;
    u64 right = (c == WHITE) ? (b << 9) & ~FileABB : (b >> 9) & ~FileHBB;
    return left | right;
}

void init_knight_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        u64 b = 1ULL << sq;
        u64 attacks = 0;
        attacks |= (b << 17) & ~FileABB;
        attacks |= (b << 15) & ~FileHBB;
        attacks |= (b << 10) & ~(FileABB | FileBBB);
        attacks |= (b << 6)  & ~(FileGBB | FileHBB);
        attacks |= (b >> 6)  & ~(FileABB | FileBBB);
        attacks |= (b >> 10) & ~(FileGBB | FileHBB);
        attacks |= (b >> 15) & ~FileABB;
        attacks |= (b >> 17) & ~FileHBB;
        KnightAttacks[sq] = attacks;
    }
}

void init_king_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        u64 b = 1ULL << sq;
        u64 attacks = 0;
        u64 k = b;
        attacks |= (k << 8);
        attacks |= (k >> 8);
        attacks |= (k << 1) & ~FileABB;
        attacks |= (k >> 1) & ~FileHBB;
        attacks |= (k << 7) & ~FileHBB;
        attacks |= (k << 9) & ~FileABB;
        attacks |= (k >> 9) & ~FileHBB;
        attacks |= (k >> 7) & ~FileABB;
        KingAttacks[sq] = attacks;
    }
}

// ============================================================
// MAGIC BITBOARDS — O(1) sliding piece attack lookup
// ============================================================

struct Magic {
    u64* attacks;
    u64 mask;
    u64 magic;
    int shift;
    int index_bits;
};

static u64 BishopAttacksTable[64][512];
static u64 RookAttacksTable[64][4096];
static Magic BishopMagics[64];
static Magic RookMagics[64];

static u64 rng_state = 1070372ULL;
static u64 magic_rand() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static u64 random_magic() {
    u64 m = 0;
    for (int i = 0; i < 6; i++)
        m |= (magic_rand() & 0xFFFFULL) << (i * 11);
    return m;
}

static u64 compute_slider_mask(int sq, bool is_rook) {
    u64 mask = 0;
    int r = rank_of(sq), f = file_of(sq);

    if (is_rook) {
        for (int i = r + 1; i < 7; i++) mask |= 1ULL << make_square(f, i);
        for (int i = r - 1; i > 0; i--) mask |= 1ULL << make_square(f, i);
        for (int i = f + 1; i < 7; i++) mask |= 1ULL << make_square(i, r);
        for (int i = f - 1; i > 0; i--) mask |= 1ULL << make_square(i, r);
    } else {
        for (int i = 1; i < 7; i++) {
            if (r + i <= 6 && f + i <= 6) mask |= 1ULL << make_square(f + i, r + i);
            if (r + i <= 6 && f - i >= 1) mask |= 1ULL << make_square(f - i, r + i);
            if (r - i >= 1 && f + i <= 6) mask |= 1ULL << make_square(f + i, r - i);
            if (r - i >= 1 && f - i >= 1) mask |= 1ULL << make_square(f - i, r - i);
        }
    }
    return mask;
}

static u64 ray_attacks_slow(int sq, u64 blockers, bool is_rook) {
    u64 attacks = 0;
    int r = rank_of(sq), f = file_of(sq);

    if (is_rook) {
        for (int d = 1; r + d <= 7; d++) { u64 b = 1ULL << make_square(f, r + d); attacks |= b; if (blockers & b) break; }
        for (int d = 1; r - d >= 0; d++) { u64 b = 1ULL << make_square(f, r - d); attacks |= b; if (blockers & b) break; }
        for (int d = 1; f + d <= 7; d++) { u64 b = 1ULL << make_square(f + d, r); attacks |= b; if (blockers & b) break; }
        for (int d = 1; f - d >= 0; d++) { u64 b = 1ULL << make_square(f - d, r); attacks |= b; if (blockers & b) break; }
    } else {
        for (int d = 1; r + d <= 7 && f + d <= 7; d++) { u64 b = 1ULL << make_square(f + d, r + d); attacks |= b; if (blockers & b) break; }
        for (int d = 1; r + d <= 7 && f - d >= 0; d++) { u64 b = 1ULL << make_square(f - d, r + d); attacks |= b; if (blockers & b) break; }
        for (int d = 1; r - d >= 0 && f + d <= 7; d++) { u64 b = 1ULL << make_square(f + d, r - d); attacks |= b; if (blockers & b) break; }
        for (int d = 1; r - d >= 0 && f - d >= 0; d++) { u64 b = 1ULL << make_square(f - d, r - d); attacks |= b; if (blockers & b) break; }
    }
    return attacks;
}

static bool try_magic(int sq, Magic& m, bool is_rook) {
    int bits = m.index_bits;
    int total = 1 << bits;
    std::fill(m.attacks, m.attacks + total, 0ULL);

    for (int i = 0; i < total; i++) {
        u64 blockers = 0;
        u64 subset = (u64)i;
        u64 mask = m.mask;
        while (subset) {
            int bit = pop_lsb(subset);
            blockers |= 1ULL << bit;
        }

        u64 attacks = ray_attacks_slow(sq, blockers, is_rook);
        u64 idx = (blockers * m.magic) >> (64 - bits);

        if (m.attacks[idx] == 0) {
            m.attacks[idx] = attacks;
        } else if (m.attacks[idx] != attacks) {
            return false;
        }
    }
    return true;
}

static void init_magics() {
    for (int sq = 0; sq < 64; sq++) {
        // Bishop
        BishopMagics[sq].mask = compute_slider_mask(sq, false);
        BishopMagics[sq].index_bits = popcount(BishopMagics[sq].mask);
        BishopMagics[sq].shift = 64 - BishopMagics[sq].index_bits;
        BishopMagics[sq].attacks = BishopAttacksTable[sq];

        for (int attempt = 0; attempt < 1000000; attempt++) {
            BishopMagics[sq].magic = random_magic();
            if (BishopMagics[sq].magic & (0xFull << 56)) continue;
            if (try_magic(sq, BishopMagics[sq], false)) break;
        }

        // Rook
        RookMagics[sq].mask = compute_slider_mask(sq, true);
        RookMagics[sq].index_bits = popcount(RookMagics[sq].mask);
        RookMagics[sq].shift = 64 - RookMagics[sq].index_bits;
        RookMagics[sq].attacks = RookAttacksTable[sq];

        for (int attempt = 0; attempt < 1000000; attempt++) {
            RookMagics[sq].magic = random_magic();
            if (try_magic(sq, RookMagics[sq], true)) break;
        }
    }
}

u64 bishop_attacks(int sq, u64 blockers) {
    const Magic& m = BishopMagics[sq];
    u64 occ = blockers & m.mask;
    return m.attacks[(occ * m.magic) >> m.shift];
}

u64 rook_attacks(int sq, u64 blockers) {
    const Magic& m = RookMagics[sq];
    u64 occ = blockers & m.mask;
    return m.attacks[(occ * m.magic) >> m.shift];
}

void init_bitboards() {
    init_knight_attacks();
    init_king_attacks();
    init_magics();
}

void print_bitboard(u64 b) {
    std::cout << "\n +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; r--) {
        std::cout << r + 1 << " |";
        for (int f = 0; f < 8; f++) {
            int sq = make_square(f, r);
            std::cout << (b & (1ULL << sq) ? " X " : " . ") << "|";
        }
        std::cout << "\n +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "   a   b   c   d   e   f   g   h\n\n";
}

} // namespace neko

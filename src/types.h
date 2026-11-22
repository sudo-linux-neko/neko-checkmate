// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#pragma once

#include <cstdint>
#include <string>
#include <array>

namespace neko {

using u64 = uint64_t;
using u32 = uint32_t;
using i32 = int32_t;

enum Color : int { WHITE, BLACK, COLOR_NB = 2 };
enum PieceType : int { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_NB = 6 };
enum Piece : int {
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_TOTAL = 12, NO_PIECE = -1
};

enum Square : int {
    A1=0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    SQ_ZERO = 0, SQ_NB = 64, NO_SQ = 64
};

enum CastlingRight : int {
    WHITE_KING  = 1, WHITE_QUEEN = 2,
    BLACK_KING  = 4, BLACK_QUEEN = 8,
    ALL_CASTLING = 15
};

enum { MAX_MOVES = 256, MAX_PLY = 128 };

struct Move {
    int from;
    int to;
    int piece;
    int captured;
    int promotion;
    int move_type; // 0=normal, 1=castle, 2=ep, 3=promo

    bool operator==(const Move& o) const {
        return from == o.from && to == o.to;
    }
    bool operator!=(const Move& o) const { return !(*this == o); }

    bool is_null() const { return from == 0 && to == 0; }
    std::string to_string() const;
};

constexpr Move NULL_MOVE{0, 0, 0, 0, 0, 0};

inline int square(int file, int rank) { return rank * 8 + file; }
inline int file_of(int sq) { return sq & 7; }
inline int rank_of(int sq) { return sq >> 3; }
inline int make_square(int file, int rank) { return rank * 8 + file; }
inline Color flip(Color c) { return Color(c ^ 1); }
inline Piece make_piece(Color c, PieceType pt) { return Piece(c * 6 + pt); }
inline Color color_of(Piece p) { return Color(p >= 6); }
inline PieceType type_of(Piece p) { return PieceType(p % 6); }

constexpr int CastlingRightArray[2] = {
    WHITE_KING | WHITE_QUEEN,
    BLACK_KING | BLACK_QUEEN
};

constexpr u64 FileABB = 0x0101010101010101ULL;
constexpr u64 FileBBB = 0x0202020202020202ULL;
constexpr u64 FileCBB = 0x0404040404040404ULL;
constexpr u64 FileDBB = 0x0808080808080808ULL;
constexpr u64 FileEBB = 0x1010101010101010ULL;
constexpr u64 FileFBB = 0x2020202020202020ULL;
constexpr u64 FileGBB = 0x4040404040404040ULL;
constexpr u64 FileHBB = 0x8080808080808080ULL;

constexpr u64 Rank1BB = 0x00000000000000FFULL;
constexpr u64 Rank2BB = 0x000000000000FF00ULL;
constexpr u64 Rank3BB = 0x0000000000FF0000ULL;
constexpr u64 Rank4BB = 0x00000000FF000000ULL;
constexpr u64 Rank5BB = 0x000000FF00000000ULL;
constexpr u64 Rank6BB = 0x0000FF0000000000ULL;
constexpr u64 Rank7BB = 0x00FF000000000000ULL;
constexpr u64 Rank8BB = 0xFF00000000000000ULL;

constexpr u64 KingSideCastleMask  = 0x9000000000000090ULL;
constexpr u64 QueenSideCastleMask = 0x1100000000000011ULL;

} // namespace neko

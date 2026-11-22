// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include "movegen.h"

namespace neko {

static void add_move(MoveList& ml, int from, int to, int piece, int captured, int promo, int type) {
    Move m;
    m.from = from;
    m.to = to;
    m.piece = piece;
    m.captured = captured;
    m.promotion = promo;
    m.move_type = type;
    ml.add(m);
}

static void generate_pawn_moves(const Board& board, MoveList& ml, Color us, bool captures_only) {
    u64 pawns = board.pieces[make_piece(us, PAWN)];
    Color them = flip(us);
    u64 enemy_occ = board.occupancy[them];
    u64 empty = ~board.all_occ;

    int push_dir = (us == WHITE) ? 8 : -8;
    int start_rank = (us == WHITE) ? 1 : 6;
    int rank7 = (us == WHITE) ? 6 : 1;

    u64 tmp = pawns;
    while (tmp) {
        int from = pop_lsb(tmp);
        int to = from + push_dir;

        if (to < 0 || to > 63) continue;

        // Push
        if (!(board.all_occ & (1ULL << to))) {
            if (!captures_only) {
                if (from / 8 == rank7) {
                    // Promotion
                    add_move(ml, from, to, make_piece(us, PAWN), NO_PIECE, make_piece(us, QUEEN), 3);
                    add_move(ml, from, to, make_piece(us, PAWN), NO_PIECE, make_piece(us, ROOK), 3);
                    add_move(ml, from, to, make_piece(us, PAWN), NO_PIECE, make_piece(us, BISHOP), 3);
                    add_move(ml, from, to, make_piece(us, PAWN), NO_PIECE, make_piece(us, KNIGHT), 3);
                } else {
                    add_move(ml, from, to, make_piece(us, PAWN), NO_PIECE, NO_PIECE, 0);
                }

                // Double push
                int to2 = from + 2 * push_dir;
                if (from / 8 == start_rank && to2 >= 0 && to2 <= 63 &&
                    !(board.all_occ & (1ULL << to)) && !(board.all_occ & (1ULL << to2))) {
                    add_move(ml, from, to2, make_piece(us, PAWN), NO_PIECE, NO_PIECE, 0);
                }
            }
        }

        // Captures
        u64 capt = pawn_attacks(from, us) & enemy_occ;
        u64 capt_bb = capt;
        while (capt_bb) {
            int to_cap = pop_lsb(capt_bb);
            Piece cap_piece = board.piece_on(to_cap);
            int cap_type = (cap_piece != NO_PIECE) ? cap_piece : make_piece(them, PAWN);

            if (to_cap / 8 == rank7) {
                add_move(ml, from, to_cap, make_piece(us, PAWN), cap_type, make_piece(us, QUEEN), 3);
                add_move(ml, from, to_cap, make_piece(us, PAWN), cap_type, make_piece(us, ROOK), 3);
                add_move(ml, from, to_cap, make_piece(us, PAWN), cap_type, make_piece(us, BISHOP), 3);
                add_move(ml, from, to_cap, make_piece(us, PAWN), cap_type, make_piece(us, KNIGHT), 3);
            } else {
                add_move(ml, from, to_cap, make_piece(us, PAWN), cap_type, NO_PIECE, 0);
            }
        }

        // En passant
        if (board.ep_square != NO_SQ) {
            u64 ep_att = pawn_attacks(from, us) & (1ULL << board.ep_square);
            if (ep_att) {
                add_move(ml, from, board.ep_square, make_piece(us, PAWN),
                         make_piece(them, PAWN), NO_PIECE, 2);
            }
        }
    }
}

static void generate_knight_moves(const Board& board, MoveList& ml, Color us, bool captures_only) {
    u64 knights = board.pieces[make_piece(us, KNIGHT)];
    u64 friendly = board.occupancy[us];

    u64 tmp = knights;
    while (tmp) {
        int from = pop_lsb(tmp);
        u64 targets = KnightAttacks[from] & ~friendly;
        u64 t = targets;
        while (t) {
            int to = pop_lsb(t);
            Piece cap = board.piece_on(to);
            if (cap != NO_PIECE || !captures_only) {
                add_move(ml, from, to, make_piece(us, KNIGHT), cap, NO_PIECE, 0);
            }
        }
    }
}

static void generate_bishop_moves(const Board& board, MoveList& ml, Color us, bool captures_only) {
    u64 bishops = board.pieces[make_piece(us, BISHOP)];
    u64 friendly = board.occupancy[us];

    u64 tmp = bishops;
    while (tmp) {
        int from = pop_lsb(tmp);
        u64 targets = bishop_attacks(from, board.all_occ) & ~friendly;
        u64 t = targets;
        while (t) {
            int to = pop_lsb(t);
            Piece cap = board.piece_on(to);
            if (cap != NO_PIECE || !captures_only) {
                add_move(ml, from, to, make_piece(us, BISHOP), cap, NO_PIECE, 0);
            }
        }
    }
}

static void generate_rook_moves(const Board& board, MoveList& ml, Color us, bool captures_only) {
    u64 rooks = board.pieces[make_piece(us, ROOK)];
    u64 friendly = board.occupancy[us];

    u64 tmp = rooks;
    while (tmp) {
        int from = pop_lsb(tmp);
        u64 targets = rook_attacks(from, board.all_occ) & ~friendly;
        u64 t = targets;
        while (t) {
            int to = pop_lsb(t);
            Piece cap = board.piece_on(to);
            if (cap != NO_PIECE || !captures_only) {
                add_move(ml, from, to, make_piece(us, ROOK), cap, NO_PIECE, 0);
            }
        }
    }
}

static void generate_queen_moves(const Board& board, MoveList& ml, Color us, bool captures_only) {
    u64 queens = board.pieces[make_piece(us, QUEEN)];
    u64 friendly = board.occupancy[us];

    u64 tmp = queens;
    while (tmp) {
        int from = pop_lsb(tmp);
        u64 targets = (bishop_attacks(from, board.all_occ) | rook_attacks(from, board.all_occ)) & ~friendly;
        u64 t = targets;
        while (t) {
            int to = pop_lsb(t);
            Piece cap = board.piece_on(to);
            if (cap != NO_PIECE || !captures_only) {
                add_move(ml, from, to, make_piece(us, QUEEN), cap, NO_PIECE, 0);
            }
        }
    }
}

static void generate_king_moves(const Board& board, MoveList& ml, Color us, bool captures_only) {
    u64 king = board.pieces[make_piece(us, KING)];
    u64 friendly = board.occupancy[us];

    if (king == 0) return;
    int from = lsb(king);
    u64 targets = KingAttacks[from] & ~friendly;
    u64 t = targets;
    while (t) {
        int to = pop_lsb(t);
        Piece cap = board.piece_on(to);
        if (cap != NO_PIECE || !captures_only) {
            add_move(ml, from, to, make_piece(us, KING), cap, NO_PIECE, 0);
        }
    }

    if (captures_only) return;

    // Castling: must not be in check, must not pass through or end on attacked square
    Color opp = flip(us);
    if (us == WHITE && from == E1) {
        // Kingside
        if ((board.castling & WHITE_KING) &&
            !(board.all_occ & ((1ULL << F1) | (1ULL << G1))) &&
            !board.is_attacked(E1, BLACK) &&
            !board.is_attacked(F1, BLACK) && !board.is_attacked(G1, BLACK)) {
            add_move(ml, E1, G1, W_KING, NO_PIECE, NO_PIECE, 1);
        }
        // Queenside
        if ((board.castling & WHITE_QUEEN) &&
            !(board.all_occ & ((1ULL << D1) | (1ULL << C1) | (1ULL << B1))) &&
            !board.is_attacked(E1, BLACK) &&
            !board.is_attacked(D1, BLACK) && !board.is_attacked(C1, BLACK)) {
            add_move(ml, E1, C1, W_KING, NO_PIECE, NO_PIECE, 1);
        }
    } else if (us == BLACK && from == E8) {
        // Kingside
        if ((board.castling & BLACK_KING) &&
            !(board.all_occ & ((1ULL << F8) | (1ULL << G8))) &&
            !board.is_attacked(E8, WHITE) &&
            !board.is_attacked(F8, WHITE) && !board.is_attacked(G8, WHITE)) {
            add_move(ml, E8, G8, B_KING, NO_PIECE, NO_PIECE, 1);
        }
        // Queenside
        if ((board.castling & BLACK_QUEEN) &&
            !(board.all_occ & ((1ULL << D8) | (1ULL << C8) | (1ULL << B8))) &&
            !board.is_attacked(E8, WHITE) &&
            !board.is_attacked(D8, WHITE) && !board.is_attacked(C8, WHITE)) {
            add_move(ml, E8, C8, B_KING, NO_PIECE, NO_PIECE, 1);
        }
    }
}

void generate_moves(const Board& board, MoveList& ml) {
    ml.count = 0;
    Color us = board.side;
    generate_pawn_moves(board, ml, us, false);
    generate_knight_moves(board, ml, us, false);
    generate_bishop_moves(board, ml, us, false);
    generate_rook_moves(board, ml, us, false);
    generate_queen_moves(board, ml, us, false);
    generate_king_moves(board, ml, us, false);
}

void generate_captures(const Board& board, MoveList& ml) {
    ml.count = 0;
    Color us = board.side;
    generate_pawn_moves(board, ml, us, true);
    generate_knight_moves(board, ml, us, true);
    generate_bishop_moves(board, ml, us, true);
    generate_rook_moves(board, ml, us, true);
    generate_queen_moves(board, ml, us, true);
    generate_king_moves(board, ml, us, true);
}

void generate_quiets(const Board& board, MoveList& ml) {
    ml.count = 0;
    // For simplicity, generate all and filter captures
    MoveList all;
    generate_moves(board, all);
    for (int i = 0; i < all.count; i++) {
        if (all.moves[i].captured == NO_PIECE && all.moves[i].move_type != 2) {
            ml.add(all.moves[i]);
        }
    }
}

} // namespace neko

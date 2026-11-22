// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include "board.h"
#include <sstream>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace neko {

static u64 ZobristPieces[12][64];
static u64 ZobristCastle[16];
static u64 ZobristEp[65];
static u64 ZobristSide;

static u64 zobrist_random() {
    static u64 state = 0x3243F6A8885A308DULL;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

void init_zobrist() {
    for (int p = 0; p < 12; p++)
        for (int s = 0; s < 64; s++)
            ZobristPieces[p][s] = zobrist_random();
    for (int c = 0; c < 16; c++)
        ZobristCastle[c] = zobrist_random();
    for (int e = 0; e < 65; e++)
        ZobristEp[e] = zobrist_random();
    ZobristSide = zobrist_random();
}

u64 compute_zobrist(const Board& b) {
    u64 key = 0;
    for (int p = 0; p < 12; p++) {
        u64 bb = b.pieces[p];
        while (bb) {
            int sq = pop_lsb(bb);
            key ^= ZobristPieces[p][sq];
        }
    }
    key ^= ZobristCastle[b.castling];
    key ^= ZobristEp[b.ep_square == NO_SQ ? 64 : b.ep_square];
    if (b.side == BLACK) key ^= ZobristSide;
    return key;
}

Piece Board::piece_on(int sq) const {
    u64 b = 1ULL << sq;
    for (int p = 0; p < 12; p++) {
        if (pieces[p] & b) return Piece(p);
    }
    return NO_PIECE;
}

void Board::set_startpos() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::set_fen(const std::string& fen) {
    memset(this, 0, sizeof(Board));
    for (int i = 0; i < 12; i++) pieces[i] = 0;
    occupancy[0] = occupancy[1] = 0;
    all_occ = 0;
    ep_square = NO_SQ;
    position_history.clear();

    std::istringstream ss(fen);
    std::string token;

    ss >> token;
    int sq = 56;
    for (char c : token) {
        if (c == '/') {
            sq -= 16;
        } else if (c >= '1' && c <= '8') {
            sq += c - '0';
        } else {
            Piece p = NO_PIECE;
            Color col = isupper(c) ? WHITE : BLACK;
            switch (tolower(c)) {
                case 'p': p = make_piece(col, PAWN); break;
                case 'n': p = make_piece(col, KNIGHT); break;
                case 'b': p = make_piece(col, BISHOP); break;
                case 'r': p = make_piece(col, ROOK); break;
                case 'q': p = make_piece(col, QUEEN); break;
                case 'k': p = make_piece(col, KING); break;
            }
            if (p != NO_PIECE) {
                pieces[p] |= 1ULL << sq;
                occupancy[col] |= 1ULL << sq;
            }
            sq++;
        }
    }

    all_occ = occupancy[0] | occupancy[1];

    ss >> token;
    side = (token == "w") ? WHITE : BLACK;

    ss >> token;
    castling = 0;
    if (token.find('K') != std::string::npos) castling |= WHITE_KING;
    if (token.find('Q') != std::string::npos) castling |= WHITE_QUEEN;
    if (token.find('k') != std::string::npos) castling |= BLACK_KING;
    if (token.find('q') != std::string::npos) castling |= BLACK_QUEEN;

    ss >> token;
    if (token != "-") {
        int f = token[0] - 'a';
        int r = token[1] - '1';
        ep_square = make_square(f, r);
    }

    ss >> halfmove;
    if (ss.eof()) halfmove = 0;

    ss >> fullmove;
    if (ss.eof()) fullmove = 1;
    if (fullmove == 0) fullmove = 1;

    zobrist = compute_zobrist(*this);
    position_history.clear();
    position_history.push_back(zobrist);
}

std::string Board::get_fen() const {
    std::string fen;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            int sq = make_square(f, r);
            Piece p = piece_on(sq);
            if (p == NO_PIECE) {
                empty++;
            } else {
                if (empty > 0) { fen += char('0' + empty); empty = 0; }
                char c;
                switch (type_of(p)) {
                    case PAWN:   c = 'p'; break;
                    case KNIGHT: c = 'n'; break;
                    case BISHOP: c = 'b'; break;
                    case ROOK:   c = 'r'; break;
                    case QUEEN:  c = 'q'; break;
                    case KING:   c = 'k'; break;
                    default:     c = '?'; break;
                }
                if (color_of(p) == WHITE) c = toupper(c);
                fen += c;
            }
        }
        if (empty > 0) fen += char('0' + empty);
        if (r > 0) fen += '/';
    }

    fen += (side == WHITE) ? " w " : " b ";

    std::string castle;
    if (castling & WHITE_KING)  castle += 'K';
    if (castling & WHITE_QUEEN) castle += 'Q';
    if (castling & BLACK_KING)  castle += 'k';
    if (castling & BLACK_QUEEN) castle += 'q';
    if (castle.empty()) castle = "-";
    fen += castle + " ";

    if (ep_square == NO_SQ)
        fen += "-";
    else
        fen += char(file_of(ep_square) + 'a') + char(rank_of(ep_square) + '1');

    fen += " " + std::to_string(halfmove) + " " + std::to_string(fullmove);
    return fen;
}

void Board::make_move(Move m) {
    int from = m.from;
    int to = m.to;
    Piece moving = piece_on(from);
    Color us = color_of(moving);
    PieceType pt = type_of(moving);

    // Remove piece from 'from'
    pieces[moving] &= ~(1ULL << from);
    occupancy[us] &= ~(1ULL << from);
    zobrist ^= ZobristPieces[moving][from];

    // Capture
    if (m.captured != NO_PIECE) {
        pieces[m.captured] &= ~(1ULL << to);
        occupancy[color_of(Piece(m.captured))] &= ~(1ULL << to);
        zobrist ^= ZobristPieces[m.captured][to];
    }

    // En passant capture
    if (m.move_type == 2) {
        int captured_sq = (us == WHITE) ? to - 8 : to + 8;
        Piece ep_pawn = make_piece(flip(us), PAWN);
        pieces[ep_pawn] &= ~(1ULL << captured_sq);
        occupancy[flip(us)] &= ~(1ULL << captured_sq);
        zobrist ^= ZobristPieces[ep_pawn][captured_sq];
    }

    // Place piece on 'to'
    PieceType promo_type = (m.promotion != NO_PIECE) ? type_of(Piece(m.promotion)) : pt;
    Piece place_piece = make_piece(us, promo_type);
    pieces[place_piece] |= 1ULL << to;
    occupancy[us] |= 1ULL << to;
    zobrist ^= ZobristPieces[place_piece][to];

    // Castling - move the rook
    if (m.move_type == 1) {
        if (to == G1) {
            pieces[W_ROOK] &= ~(1ULL << H1);
            pieces[W_ROOK] |= 1ULL << F1;
            occupancy[WHITE] &= ~(1ULL << H1);
            occupancy[WHITE] |= 1ULL << F1;
            zobrist ^= ZobristPieces[W_ROOK][H1] ^ ZobristPieces[W_ROOK][F1];
        } else if (to == C1) {
            pieces[W_ROOK] &= ~(1ULL << A1);
            pieces[W_ROOK] |= 1ULL << D1;
            occupancy[WHITE] &= ~(1ULL << A1);
            occupancy[WHITE] |= 1ULL << D1;
            zobrist ^= ZobristPieces[W_ROOK][A1] ^ ZobristPieces[W_ROOK][D1];
        } else if (to == G8) {
            pieces[B_ROOK] &= ~(1ULL << H8);
            pieces[B_ROOK] |= 1ULL << F8;
            occupancy[BLACK] &= ~(1ULL << H8);
            occupancy[BLACK] |= 1ULL << F8;
            zobrist ^= ZobristPieces[B_ROOK][H8] ^ ZobristPieces[B_ROOK][F8];
        } else if (to == C8) {
            pieces[B_ROOK] &= ~(1ULL << A8);
            pieces[B_ROOK] |= 1ULL << D8;
            occupancy[BLACK] &= ~(1ULL << A8);
            occupancy[BLACK] |= 1ULL << D8;
            zobrist ^= ZobristPieces[B_ROOK][A8] ^ ZobristPieces[B_ROOK][D8];
        }
    }

    // Update castling rights
    zobrist ^= ZobristCastle[castling];
    if (pt == KING) {
        if (us == WHITE) castling &= ~(WHITE_KING | WHITE_QUEEN);
        else             castling &= ~(BLACK_KING | BLACK_QUEEN);
    }
    if (from == A1 || to == A1) castling &= ~WHITE_QUEEN;
    if (from == H1 || to == H1) castling &= ~WHITE_KING;
    if (from == A8 || to == A8) castling &= ~BLACK_QUEEN;
    if (from == H8 || to == H8) castling &= ~BLACK_KING;
    zobrist ^= ZobristCastle[castling];

    // En passant square
    zobrist ^= ZobristEp[ep_square == NO_SQ ? 64 : ep_square];
    ep_square = NO_SQ;
    if (pt == PAWN) {
        if (rank_of(to) - rank_of(from) == 2) {
            ep_square = make_square(file_of(from), rank_of(from) + 1);
        } else if (rank_of(from) - rank_of(to) == 2) {
            ep_square = make_square(file_of(from), rank_of(from) - 1);
        }
    }
    zobrist ^= ZobristEp[ep_square == NO_SQ ? 64 : ep_square];

    // Halfmove clock
    if (pt == PAWN || m.captured != NO_PIECE || m.move_type == 2)
        halfmove = 0;
    else
        halfmove++;

    // Fullmove number
    if (side == BLACK) fullmove++;

    // Side to move
    zobrist ^= ZobristSide;
    side = flip(side);
    zobrist ^= ZobristSide;

    all_occ = occupancy[0] | occupancy[1];

    // Track position for repetition detection
    position_history.push_back(zobrist);
}

void Board::undo_move(Move m, const UndoInfo& u) {
    // Restore side to move first
    zobrist ^= ZobristSide;
    side = flip(side);
    zobrist ^= ZobristSide;

    int from = m.from;
    int to = m.to;
    Color us = color_of(piece_on(to));
    PieceType pt = type_of(Piece(m.piece));

    // Remove piece from 'to'
    PieceType promo_type = (m.promotion != NO_PIECE) ? type_of(Piece(m.promotion)) : pt;
    Piece place_piece = make_piece(us, promo_type);
    pieces[place_piece] &= ~(1ULL << to);
    occupancy[us] &= ~(1ULL << to);
    zobrist ^= ZobristPieces[place_piece][to];

    // Restore captured piece (if any, but NOT en passant — EP captures restore via the block below)
    if (m.captured != NO_PIECE && m.move_type != 2) {
        pieces[m.captured] |= 1ULL << to;
        occupancy[color_of(Piece(m.captured))] |= 1ULL << to;
        zobrist ^= ZobristPieces[m.captured][to];
    }

    // En passant undo
    if (m.move_type == 2) {
        int captured_sq = (us == WHITE) ? to - 8 : to + 8;
        Piece ep_pawn = make_piece(flip(us), PAWN);
        pieces[ep_pawn] |= 1ULL << captured_sq;
        occupancy[flip(us)] |= 1ULL << captured_sq;
        zobrist ^= ZobristPieces[ep_pawn][captured_sq];
    }

    // Place piece back on 'from'
    Piece orig_piece = make_piece(us, pt);
    pieces[orig_piece] |= 1ULL << from;
    occupancy[us] |= 1ULL << from;
    zobrist ^= ZobristPieces[orig_piece][from];

    // Castling undo - move rook back
    if (m.move_type == 1) {
        if (to == G1) {
            pieces[W_ROOK] &= ~(1ULL << F1);
            pieces[W_ROOK] |= 1ULL << H1;
            occupancy[WHITE] &= ~(1ULL << F1);
            occupancy[WHITE] |= 1ULL << H1;
            zobrist ^= ZobristPieces[W_ROOK][F1] ^ ZobristPieces[W_ROOK][H1];
        } else if (to == C1) {
            pieces[W_ROOK] &= ~(1ULL << D1);
            pieces[W_ROOK] |= 1ULL << A1;
            occupancy[WHITE] &= ~(1ULL << D1);
            occupancy[WHITE] |= 1ULL << A1;
            zobrist ^= ZobristPieces[W_ROOK][D1] ^ ZobristPieces[W_ROOK][A1];
        } else if (to == G8) {
            pieces[B_ROOK] &= ~(1ULL << F8);
            pieces[B_ROOK] |= 1ULL << H8;
            occupancy[BLACK] &= ~(1ULL << F8);
            occupancy[BLACK] |= 1ULL << H8;
            zobrist ^= ZobristPieces[B_ROOK][F8] ^ ZobristPieces[B_ROOK][H8];
        } else if (to == C8) {
            pieces[B_ROOK] &= ~(1ULL << D8);
            pieces[B_ROOK] |= 1ULL << A8;
            occupancy[BLACK] &= ~(1ULL << D8);
            occupancy[BLACK] |= 1ULL << A8;
            zobrist ^= ZobristPieces[B_ROOK][D8] ^ ZobristPieces[B_ROOK][A8];
        }
    }

    // Restore state
    castling = u.castling;
    ep_square = u.ep_square;
    halfmove = u.halfmove;
    zobrist = u.zobrist;
    all_occ = occupancy[0] | occupancy[1];

    // Undo position tracking
    if (!position_history.empty())
        position_history.pop_back();
}

void Board::make_null_move() {
    zobrist ^= ZobristEp[ep_square == NO_SQ ? 64 : ep_square];
    ep_square = NO_SQ;
    zobrist ^= ZobristSide;
    side = flip(side);
    zobrist ^= ZobristSide;
}

void Board::undo_null_move(int old_ep, u64 old_zobrist) {
    side = flip(side);
    ep_square = old_ep;
    zobrist = old_zobrist;
}

Piece Board::piece_at(int sq) const {
    return piece_on(sq);
}

bool Board::is_attacked(int sq, Color by) const {
    // Pawn attacks
    u64 pawns = pieces[make_piece(by, PAWN)];
    u64 pawn_att = 0;
    if (by == WHITE) {
        pawn_att = ((pawns << 7) & ~FileHBB) | ((pawns << 9) & ~FileABB);
    } else {
        pawn_att = ((pawns >> 7) & ~FileABB) | ((pawns >> 9) & ~FileHBB);
    }
    if (pawn_att & (1ULL << sq)) return true;

    // Knight attacks
    if (KnightAttacks[sq] & pieces[make_piece(by, KNIGHT)]) return true;

    // King attacks
    if (KingAttacks[sq] & pieces[make_piece(by, KING)]) return true;

    // Bishop/Queen attacks (diagonals)
    u64 diag_sliders = pieces[make_piece(by, BISHOP)] | pieces[make_piece(by, QUEEN)];
    if (bishop_attacks(sq, all_occ) & diag_sliders) return true;

    // Rook/Queen attacks (straights)
    u64 str_sliders = pieces[make_piece(by, ROOK)] | pieces[make_piece(by, QUEEN)];
    if (rook_attacks(sq, all_occ) & str_sliders) return true;

    return false;
}

bool Board::in_check() const {
    Piece king = make_piece(side, KING);
    u64 king_bb = pieces[king];
    if (king_bb == 0) return false;
    int king_sq = lsb(king_bb);
    return is_attacked(king_sq, flip(side));
}

int Board::repetitions() const {
    int count = 0;
    // Only search back to the last irreversible move (pawn move or capture)
    // to avoid quadratic blowup and false repetitions.
    int limit = std::max(0, (int)position_history.size() - halfmove);
    for (int i = (int)position_history.size() - 1; i >= limit; i--) {
        if (position_history[i] == zobrist)
            count++;
    }
    return count;
}

u64 Board::get_piece_attacks(int sq) const {
    Piece p = piece_on(sq);
    if (p == NO_PIECE) return 0;
    Color c = color_of(p);
    PieceType pt = type_of(p);
    switch (pt) {
        case PAWN:   return pawn_attacks(sq, c);
        case KNIGHT: return KnightAttacks[sq];
        case KING:   return KingAttacks[sq];
        case BISHOP: return bishop_attacks(sq, all_occ);
        case ROOK:   return rook_attacks(sq, all_occ);
        case QUEEN:  return bishop_attacks(sq, all_occ) | rook_attacks(sq, all_occ);
        default:     return 0;
    }
}

void Board::print() const {
    std::string piece_chars = "PNBRQKpnbrqk";
    std::cout << "\n +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; r--) {
        std::cout << r + 1 << " |";
        for (int f = 0; f < 8; f++) {
            int sq = make_square(f, r);
            Piece p = piece_on(sq);
            if (p == NO_PIECE)
                std::cout << "   |";
            else
                std::cout << " " << piece_chars[p] << " |";
        }
        std::cout << "\n +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "   a   b   c   d   e   f   g   h\n";
    std::cout << "FEN: " << get_fen() << "\n\n";
}

} // namespace neko

// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include "search.h"
#include "evaluate.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <cstring>

namespace neko {

TranspositionTable TT;

static u64 now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

void TranspositionTable::init() {
    table.resize(TT_SIZE);
    clear();
    age = 0;
}

void TranspositionTable::clear() {
    std::fill(table.begin(), table.end(), TTEntry{0, 0, 0, 0, NULL_MOVE, 0});
    age = 0;
}

bool TranspositionTable::probe(u64 key, TTEntry& entry) {
    entry = table[key & (TT_SIZE - 1)];
    return entry.key == key && entry.depth != 0;
}

void TranspositionTable::prefetch(u64 key) const {
    __builtin_prefetch(&table[key & (TT_SIZE - 1)], 0, 3);
}

void TranspositionTable::store(u64 key, int depth, int score, int flag, Move best_move) {
    TTEntry& entry = table[key & (TT_SIZE - 1)];
    if (entry.key != key || depth >= entry.depth || entry.age < age) {
        entry.key = key;
        entry.depth = depth;
        entry.score = score;
        entry.flag = flag;
        entry.best_move = best_move;
        entry.age = age;
    }
}

void SearchInfo::clear() {
    stopped = false;
    nodes = 0;
    max_depth = 0;
    time_limit_ms = 0;
    start_time = 0;
    sel_depth = 0;
    for (auto& k : killers) k = {NULL_MOVE, NULL_MOVE};
    for (auto& h : history) h.fill(0);
}

// ============================================================
// SEE (Static Exchange Evaluation)
// ============================================================
static constexpr int SEE_PIECE_VAL[6] = { 100, 320, 330, 500, 900, 0 };

int see(const Board& board, Move m) {
    if (m.move_type == 2) return 0;
    if (m.captured == NO_PIECE && m.promotion == NO_PIECE) return 0;

    int from = m.from;
    int to = m.to;
    int victim_val = (m.captured != NO_PIECE) ? SEE_PIECE_VAL[type_of(Piece(m.captured))] : 0;
    int attacker_val = SEE_PIECE_VAL[type_of(Piece(m.piece))];
    int gain[32];
    int depth = 0;

    gain[0] = victim_val;

    u64 occ = board.all_occ;
    u64 sq_bb = 1ULL << to;

    u64 attackers =
        (pawn_attacks(to, WHITE) & board.pieces[make_piece(BLACK, PAWN)]) |
        (pawn_attacks(to, BLACK) & board.pieces[make_piece(WHITE, PAWN)]) |
        (KnightAttacks[to] & (board.pieces[make_piece(WHITE, KNIGHT)] | board.pieces[make_piece(BLACK, KNIGHT)])) |
        (bishop_attacks(to, occ) & (board.pieces[make_piece(WHITE, BISHOP)] | board.pieces[make_piece(BLACK, BISHOP)] |
                                    board.pieces[make_piece(WHITE, QUEEN)]  | board.pieces[make_piece(BLACK, QUEEN)])) |
        (rook_attacks(to, occ) & (board.pieces[make_piece(WHITE, ROOK)] | board.pieces[make_piece(BLACK, ROOK)] |
                                  board.pieces[make_piece(WHITE, QUEEN)] | board.pieces[make_piece(BLACK, QUEEN)]));

    attackers &= occ;

    occ ^= (1ULL << from);
    if (m.captured != NO_PIECE)
        occ ^= sq_bb;

    attackers |= (bishop_attacks(to, occ) & (board.pieces[make_piece(WHITE, BISHOP)] | board.pieces[make_piece(BLACK, BISHOP)] |
                                             board.pieces[make_piece(WHITE, QUEEN)]  | board.pieces[make_piece(BLACK, QUEEN)])) &
                 ~attackers;
    attackers |= (rook_attacks(to, occ) & (board.pieces[make_piece(WHITE, ROOK)] | board.pieces[make_piece(BLACK, ROOK)] |
                                           board.pieces[make_piece(WHITE, QUEEN)] | board.pieces[make_piece(BLACK, QUEEN)])) &
                ~attackers;
    attackers &= occ;

    Color side = color_of(Piece(m.piece));
    PieceType pt = type_of(Piece(m.piece));

    while (true) {
        depth++;
        gain[depth] = attacker_val - gain[depth - 1];
        if (std::max(-gain[depth - 1], gain[depth]) < 0) break;

        attackers ^= sq_bb;
        occ ^= sq_bb;

        side = flip(side);

        u64 new_diag = bishop_attacks(to, occ) & (board.pieces[make_piece(WHITE, BISHOP)] | board.pieces[make_piece(BLACK, BISHOP)] |
                                                  board.pieces[make_piece(WHITE, QUEEN)]  | board.pieces[make_piece(BLACK, QUEEN)]);
        u64 new_straight = rook_attacks(to, occ) & (board.pieces[make_piece(WHITE, ROOK)] | board.pieces[make_piece(BLACK, ROOK)] |
                                                    board.pieces[make_piece(WHITE, QUEEN)] | board.pieces[make_piece(BLACK, QUEEN)]);
        attackers |= (new_diag | new_straight) & ~attackers;
        attackers &= occ;

        if (depth >= 31) break;

        u64 friendly_attackers;
        if (side == WHITE)
            friendly_attackers = attackers & (board.pieces[make_piece(WHITE, PAWN)] | board.pieces[make_piece(WHITE, KNIGHT)] |
                                              board.pieces[make_piece(WHITE, BISHOP)] | board.pieces[make_piece(WHITE, ROOK)] |
                                              board.pieces[make_piece(WHITE, QUEEN)] | board.pieces[make_piece(WHITE, KING)]);
        else
            friendly_attackers = attackers & (board.pieces[make_piece(BLACK, PAWN)] | board.pieces[make_piece(BLACK, KNIGHT)] |
                                              board.pieces[make_piece(BLACK, BISHOP)] | board.pieces[make_piece(BLACK, ROOK)] |
                                              board.pieces[make_piece(BLACK, QUEEN)] | board.pieces[make_piece(BLACK, KING)]);

        if (!friendly_attackers) break;

        if (side == WHITE) {
            if (friendly_attackers & board.pieces[make_piece(WHITE, PAWN)])       { pt = PAWN;   attacker_val = SEE_PIECE_VAL[PAWN]; }
            else if (friendly_attackers & board.pieces[make_piece(WHITE, KNIGHT)]) { pt = KNIGHT; attacker_val = SEE_PIECE_VAL[KNIGHT]; }
            else if (friendly_attackers & board.pieces[make_piece(WHITE, BISHOP)]) { pt = BISHOP; attacker_val = SEE_PIECE_VAL[BISHOP]; }
            else if (friendly_attackers & board.pieces[make_piece(WHITE, ROOK)])   { pt = ROOK;   attacker_val = SEE_PIECE_VAL[ROOK]; }
            else                                                                    { pt = QUEEN;  attacker_val = SEE_PIECE_VAL[QUEEN]; }
        } else {
            if (friendly_attackers & board.pieces[make_piece(BLACK, PAWN)])       { pt = PAWN;   attacker_val = SEE_PIECE_VAL[PAWN]; }
            else if (friendly_attackers & board.pieces[make_piece(BLACK, KNIGHT)]) { pt = KNIGHT; attacker_val = SEE_PIECE_VAL[KNIGHT]; }
            else if (friendly_attackers & board.pieces[make_piece(BLACK, BISHOP)]) { pt = BISHOP; attacker_val = SEE_PIECE_VAL[BISHOP]; }
            else if (friendly_attackers & board.pieces[make_piece(BLACK, ROOK)])   { pt = ROOK;   attacker_val = SEE_PIECE_VAL[ROOK]; }
            else                                                                    { pt = QUEEN;  attacker_val = SEE_PIECE_VAL[QUEEN]; }
        }
    }

    while (--depth > 0) {
        gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
    }
    return gain[0];
}

int see_sign(const Board& board, Move m) {
    if (m.captured == NO_PIECE) return 0;
    if (m.move_type == 2) return 1;
    int val = SEE_PIECE_VAL[type_of(Piece(m.captured))] - SEE_PIECE_VAL[type_of(Piece(m.piece))];
    return (val >= 0) ? 1 : val;
}

// ============================================================
// MOVE ORDERING
// ============================================================

static int mvv_lva(const Board& board, Move m) {
    int victim_score[6] = { 100, 320, 330, 500, 900, 0 };
    int attacker_score[6] = { 100, 320, 330, 500, 900, 0 };
    int victim = (m.captured != NO_PIECE) ? type_of(Piece(m.captured)) : -1;
    int attacker = type_of(Piece(m.piece));
    if (victim >= 0)
        return victim_score[victim] * 10 - attacker_score[attacker];
    return 0;
}

void order_moves(const Board& board, MoveList& ml, const TTEntry& tt,
                 const SearchInfo& info, int ply) {
    for (int i = 0; i < ml.count; i++) {
        Move& m = ml.moves[i];
        ml.scores[i] = 0;
        if (tt.best_move == m) { ml.scores[i] = 10000000; continue; }
        if (m.captured != NO_PIECE) { ml.scores[i] = 1000000 + mvv_lva(board, m); continue; }
        if (ply < MAX_PLY) {
            if (info.killers[ply][0] == m) { ml.scores[i] = 900000; continue; }
            if (info.killers[ply][1] == m) { ml.scores[i] = 800000; continue; }
        }
        ml.scores[i] = info.history[m.from][m.to];
    }
    for (int i = 1; i < ml.count; i++) {
        for (int j = i; j > 0 && ml.scores[j] > ml.scores[j-1]; j--) {
            std::swap(ml.moves[j], ml.moves[j-1]);
            std::swap(ml.scores[j], ml.scores[j-1]);
        }
    }
}

// ============================================================
// QUIESCENCE SEARCH
// ============================================================

static constexpr int MAX_QS_PLY = 10;

int quiescence(Board& board, NNUEState& nnue, SearchInfo& info,
               int alpha, int beta, int ply, int qs_ply) {
    if (info.should_stop()) return 0;
    if (ply >= MAX_PLY - 1 || qs_ply >= MAX_QS_PLY) return evaluate(board, nnue);
    info.nodes++;

    if ((info.nodes & 4095) == 0) {
        if (now_ms() - info.start_time >= (u64)info.time_limit_ms)
            info.stopped = true;
        if (info.shared_stopped && info.shared_stopped->load(std::memory_order_relaxed))
            info.stopped = true;
    }

    if (ply > info.sel_depth) info.sel_depth = ply;

    bool in_check = board.in_check();

    // When in check: generate ALL legal moves (evasions), not just captures.
    // This is critical for detecting checkmates and finding mating sequences.
    if (in_check) {
        MoveList ml;
        generate_moves(board, ml);

        if (ml.count == 0) return -30000 + ply;

        int best = -32000;
        int legal_found = 0;
        for (int i = 0; i < ml.count; i++) {
            Move m = ml.moves[i];

            UndoInfo u;
            u.castling = board.castling;
            u.ep_square = board.ep_square;
            u.halfmove = board.halfmove;
            u.zobrist = board.zobrist;
            u.captured_piece = m.captured;

            nnue.push();
            board.make_move(m);
            nnue.update_accumulator(board, m);

            Color us = flip(board.side);
            Piece king = make_piece(us, KING);
            bool illegal = (board.pieces[king] == 0) || board.is_attacked(lsb(board.pieces[king]), board.side);

            int score;
            if (illegal) {
                score = -32000;
            } else {
                score = -quiescence(board, nnue, info, -beta, -alpha, ply + 1, qs_ply + 1);
            }

            board.undo_move(m, u);
            nnue.pop();

            if (illegal) continue;
            if (info.should_stop()) return 0;
            legal_found++;
            if (score > best) best = score;
            if (score > alpha) alpha = score;
            if (alpha >= beta) return beta;
        }
        if (legal_found == 0) return -30000 + ply;
        return best;
    }

    int stand_pat = evaluate(board, nnue);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    MoveList ml;
    generate_captures(board, ml);

    for (int i = 0; i < ml.count; i++)
        ml.scores[i] = mvv_lva(board, ml.moves[i]);
    for (int i = 1; i < ml.count; i++) {
        for (int j = i; j > 0 && ml.scores[j] > ml.scores[j-1]; j--) {
            std::swap(ml.moves[j], ml.moves[j-1]);
            std::swap(ml.scores[j], ml.scores[j-1]);
        }
    }

    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];

        if (type_of(Piece(m.piece)) != KING) {
            int cap_value = (m.captured != NO_PIECE) ? SEE_PIECE_VAL[type_of(Piece(m.captured))] : 0;
            if (stand_pat + cap_value + 200 < alpha) continue;
            if (see(board, m) < 0) continue;
        }

        UndoInfo u;
        u.castling = board.castling;
        u.ep_square = board.ep_square;
        u.halfmove = board.halfmove;
        u.zobrist = board.zobrist;
        u.captured_piece = m.captured;

        nnue.push();
        board.make_move(m);
        nnue.update_accumulator(board, m);

        Color us = flip(board.side);
        Piece king = make_piece(us, KING);
        bool illegal = (board.pieces[king] == 0) || board.is_attacked(lsb(board.pieces[king]), board.side);

        int score;
        if (illegal) {
            score = alpha;
        } else {
            score = -quiescence(board, nnue, info, -beta, -alpha, ply + 1, qs_ply + 1);
        }

        board.undo_move(m, u);
        nnue.pop();

        if (illegal) continue;
        if (info.should_stop()) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ============================================================
// ALPHA-BETA SEARCH with all optimizations
// ============================================================

int alpha_beta(Board& board, NNUEState& nnue, SearchInfo& info,
               int alpha, int beta, int depth, int ply) {
    if (info.should_stop()) return 0;
    info.nodes++;

    if ((info.nodes & 4095) == 0) {
        if (now_ms() - info.start_time >= (u64)info.time_limit_ms)
            info.stopped = true;
        if (info.shared_stopped && info.shared_stopped->load(std::memory_order_relaxed))
            info.stopped = true;
    }

    if (board.halfmove >= 100) return 0;
    if (ply > 0 && board.repetitions() >= 2) return 0;

    if (depth <= 0) {
        if (ply >= MAX_PLY - 1) return evaluate(board, nnue);
        return quiescence(board, nnue, info, alpha, beta, ply, 0);
    }

    if (ply > info.sel_depth) info.sel_depth = ply;

    bool in_check = board.in_check();
    if (in_check) depth++;

    // Null move pruning — do NOT update accumulator for null moves.
    // Null move only changes side_to_move and ep_square; piece placement
    // is unchanged so the accumulator remains valid. Push/pop saves/restores it.
    if (!in_check && depth >= 3 && ply > 0) {
        Color us = board.side;
        u64 major_minor = board.pieces[make_piece(us, KNIGHT)] |
                          board.pieces[make_piece(us, BISHOP)] |
                          board.pieces[make_piece(us, ROOK)] |
                          board.pieces[make_piece(us, QUEEN)];
        if (major_minor) {
            int R = 3;
            if (depth >= 6) R = 4;

            int old_ep = board.ep_square;
            u64 old_zobrist = board.zobrist;

            nnue.push();
            board.make_null_move();
            // BUG FIX: Do NOT call update_accumulator for null moves.
            // NULL_MOVE.captured == 0 == W_PWN, which corrupts the accumulator.

            TT.prefetch(board.zobrist);

            int score = -alpha_beta(board, nnue, info, -beta, -beta + 1, depth - R, ply + 1);

            board.undo_null_move(old_ep, old_zobrist);
            nnue.pop();

            if (info.should_stop()) return 0;
            if (score >= beta) return beta;
        }
    }

    // Futility pruning guard
    bool fprune = false;
    int futility_margin = 0;
    if (!in_check && depth <= 2 && ply > 0) {
        int eval = evaluate(board, nnue);
        if (depth == 1) {
            futility_margin = 150;
            if (eval + futility_margin <= alpha) fprune = true;
        } else {
            futility_margin = 300;
            if (eval + futility_margin <= alpha) fprune = true;
        }
    }

    // TT probe
    TTEntry tt_entry;
    bool tt_hit = TT.probe(board.zobrist, tt_entry);
    Move tt_move = tt_hit ? tt_entry.best_move : NULL_MOVE;

    if (tt_hit && tt_entry.depth >= depth) {
        int tt_score = tt_entry.score;
        if (tt_score > 29000) tt_score += ply;
        else if (tt_score < -29000) tt_score -= ply;
        if (tt_entry.flag == 0) return tt_score;
        if (tt_entry.flag == 1 && tt_score <= alpha) return tt_score;
        if (tt_entry.flag == 2 && tt_score >= beta) return tt_score;
    }

    MoveList ml;
    generate_moves(board, ml);

    if (ml.count == 0) {
        if (in_check) return -30000 + ply;
        return 0;
    }

    order_moves(board, ml, tt_entry, info, ply);

    int orig_alpha = alpha;
    int best_score = -32000;
    Move best_move = ml.moves[0];
    int moves_searched = 0;
    int legal_moves_found = 0;

    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];

        UndoInfo u;
        u.castling = board.castling;
        u.ep_square = board.ep_square;
        u.halfmove = board.halfmove;
        u.zobrist = board.zobrist;
        u.captured_piece = m.captured;

        // Push accumulator state BEFORE making the move
        nnue.push();
        board.make_move(m);
        nnue.update_accumulator(board, m);

        // Detect if this move gives check (critical for LMR and mate detection)
        bool gives_check = board.in_check();

        Color us = flip(board.side);
        Piece king = make_piece(us, KING);
        bool illegal = (board.pieces[king] == 0) || board.is_attacked(lsb(board.pieces[king]), board.side);

        if (illegal) {
            board.undo_move(m, u);
            nnue.pop();
            continue;
        }

        legal_moves_found++;

        // Futility pruning — skip quiet moves that can't possibly raise alpha
        if (fprune && moves_searched > 0 && m.captured == NO_PIECE && m.promotion == NO_PIECE) {
            board.undo_move(m, u);
            nnue.pop();
            continue;
        }

        // Prefetch TT for child position
        TT.prefetch(board.zobrist);

        int score;

        // PVS + LMR
        if (moves_searched == 0) {
            score = -alpha_beta(board, nnue, info, -beta, -alpha, depth - 1, ply + 1);
        } else {
            // LMR — never reduce checking moves (they are tactically dangerous)
            int reduction = 0;
            if (depth >= 3 && moves_searched >= 3 && !in_check && !gives_check &&
                m.captured == NO_PIECE && m.promotion == NO_PIECE) {
                reduction = 1;
                if (moves_searched >= 6) reduction = 2;
                if (moves_searched >= 12) reduction = 3;
                reduction = std::min(reduction, depth - 2);

                // Reduce less for killer moves
                if (ply < MAX_PLY && (info.killers[ply][0] == m || info.killers[ply][1] == m))
                    reduction--;
            }

            // Null-window search with LMR
            score = -alpha_beta(board, nnue, info, -alpha - 1, -alpha, depth - 1 - reduction, ply + 1);

            // Re-search at full depth if reduced search found something interesting
            if (reduction > 0 && score > alpha) {
                score = -alpha_beta(board, nnue, info, -alpha - 1, -alpha, depth - 1, ply + 1);
            }

            // PVS re-search with full window
            if (score > alpha && score < beta) {
                score = -alpha_beta(board, nnue, info, -beta, -alpha, depth - 1, ply + 1);
            }
        }

        board.undo_move(m, u);
        nnue.pop();

        if (info.should_stop()) return 0;
        moves_searched++;

        if (score > best_score) {
            best_score = score;
            best_move = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            if (m.captured == NO_PIECE && m.promotion == NO_PIECE) {
                if (ply < MAX_PLY) {
                    info.killers[ply][1] = info.killers[ply][0];
                    info.killers[ply][0] = m;
                }
                info.history[m.from][m.to] += depth * depth;
                if (info.history[m.from][m.to] > 500000000)
                    for (auto& h : info.history) h.fill(0);
            }
            break;
        }
    }

    if (legal_moves_found == 0) {
        if (in_check) return -30000 + ply;
        return 0;
    }

    int flag = (best_score <= orig_alpha) ? 1 : (best_score >= beta) ? 2 : 0;
    int tt_score = best_score;
    if (tt_score > 29000) tt_score -= ply;
    else if (tt_score < -29000) tt_score += ply;
    TT.store(board.zobrist, depth, tt_score, flag, best_move);

    return best_score;
}

// ============================================================
// ITERATIVE DEEPENING with aspiration windows
// ============================================================

SearchResult iterative_deepening(Board& board, NNUEState& nnue, SearchInfo& info) {
    SearchResult result = {NULL_MOVE, 0, 0, 0};
    info.start_time = now_ms();
    info.stopped = false;
    info.nodes = 0;
    info.sel_depth = 0;
    TT.age++;

    // Full reset at start of search — initializes accumulator from scratch
    nnue.reset(board);

    int prev_score = 0;

    for (int depth = 1; depth <= info.max_depth; depth++) {
        int alpha = -30000, beta = 30000;

        if (depth >= 5) {
            int window = 25;
            alpha = prev_score - window;
            beta = prev_score + window;

            while (true) {
                int score = alpha_beta(board, nnue, info, alpha, beta, depth, 0);

                if (info.should_stop()) break;

                if (score <= alpha) {
                    alpha = std::max(alpha - window * 4, -30000);
                } else if (score >= beta) {
                    beta = std::min(beta + window * 4, 30000);
                } else {
                    prev_score = score;
                    break;
                }
                window *= 4;
            }
        } else {
            prev_score = alpha_beta(board, nnue, info, alpha, beta, depth, 0);
        }

        if (!info.should_stop()) {
            TTEntry entry;
            if (TT.probe(board.zobrist, entry)) {
                result.best_move = entry.best_move;
            } else if (depth == 1) {
                MoveList ml;
                generate_moves(board, ml);
                if (ml.count > 0) result.best_move = ml.moves[0];
            }
            result.score = prev_score;
            result.depth = depth;
            result.nodes = info.nodes;

            u64 elapsed = now_ms() - info.start_time;
            if (elapsed == 0) elapsed = 1;
            std::cout << "info depth " << depth
                      << " seldepth " << info.sel_depth
                      << " score cp " << prev_score
                      << " nodes " << info.nodes
                      << " time " << elapsed
                      << " nps " << (info.nodes * 1000ULL / elapsed)
                      << " pv "
                      << "\n";
        }

        u64 elapsed = now_ms() - info.start_time;
        if (elapsed >= (u64)info.time_limit_ms / 2) break;
    }

    return result;
}

} // namespace neko

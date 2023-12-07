use std::cmp::{max, min};
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::board::board::Board;
use crate::engine::transposition::EntryFlag;
use crate::moves::movegenerator::MGT;
use crate::moves::movelist::{MoveListEntry, BAD_CAPTURE};
use crate::moves::moves::Move;
use crate::search::{SearchStack, INIT_ASP};

use super::history_heuristics::MAX_HIST_VAL;
use super::quiescence::quiescence;
use super::thread::ThreadData;
use super::{
    get_reduction, store_pv, SearchType, LMP_CONST, LMR_THRESHOLD, MAX_LMP_DEPTH, MAX_RFP_DEPTH, MIN_IIR_DEPTH,
    MIN_LMR_DEPTH, MIN_NMP_DEPTH, RFP_MULTIPLIER,
};

pub const CHECKMATE: i32 = 30000;
pub const STALEMATE: i32 = 0;
pub const NEAR_CHECKMATE: i32 = CHECKMATE - 1000;
pub const INFINITY: i32 = 50000;
pub const MAX_SEARCH_DEPTH: i32 = 100;

pub fn search(td: &mut ThreadData, print_uci: bool, board: Board) -> Move {
    td.game_time.search_start = Instant::now();
    td.root_color = board.to_move;
    td.nodes_searched = 0;
    td.stack = SearchStack::default();

    let best_move = iterative_deepening(td, &board, print_uci);

    assert_ne!(best_move, Move::NULL);

    best_move
}

pub(crate) fn iterative_deepening(td: &mut ThreadData, board: &Board, print_uci: bool) -> Move {
    let mut pv = Vec::new();
    let mut best_move = Move::NULL;
    let mut prev_score = -INFINITY;

    for depth in 1..=td.max_depth {
        td.iter_max_depth = depth;
        td.sel_depth = 0;

        let score = aspiration_windows(td, &mut pv, prev_score, board);
        prev_score = score;

        if !pv.is_empty() {
            best_move = pv[0];
        }

        if print_uci {
            td.print_search_stats(score, &pv);
        }

        if td.search_type == SearchType::Time && td.game_time.soft_termination() {
            break;
        }
        if td.halt.load(Ordering::SeqCst) {
            break;
        }
    }

    assert_ne!(best_move, Move::NULL);
    best_move
}

fn aspiration_windows(td: &mut ThreadData, pv: &mut Vec<Move>, prev_score: i32, board: &Board) -> i32 {
    let mut alpha = -INFINITY;
    let mut beta = INFINITY;
    let mut delta = INIT_ASP + prev_score * prev_score / 10000;

    if td.iter_max_depth >= 4 {
        alpha = alpha.max(prev_score - delta);
        beta = beta.min(prev_score + delta);
    }

    loop {
        let score = alpha_beta::<true>(td.iter_max_depth, alpha, beta, pv, td, board, false);
        if score <= alpha {
            beta = (alpha + beta) / 2;
            alpha = max(score - delta, -INFINITY);
        } else if score >= beta {
            beta = min(score + delta, INFINITY);
        } else {
            return score;
        }
        delta += delta / 3;
    }
}

/// Principal variation search - uses reduced alpha beta windows around a likely best move candidate
/// to refute other variations
#[allow(clippy::too_many_arguments)]
fn alpha_beta<const IS_PV: bool>(
    mut depth: i32,
    mut alpha: i32,
    beta: i32,
    pv: &mut Vec<Move>,
    td: &mut ThreadData,
    board: &Board,
    cut_node: bool,
) -> i32 {
    let ply = td.iter_max_depth - depth;
    let is_root = ply == 0;
    let in_check = board.in_check;
    td.sel_depth = td.sel_depth.max(ply);

    if td.halt.load(Ordering::Relaxed) || td.game_time.hard_termination() {
        td.halt.store(true, Ordering::SeqCst);
        // return board.evaluate();
        return 0;
    }

    // Needed since the function can calculate extensions in cases where it finds itself in check
    if ply >= MAX_SEARCH_DEPTH {
        if board.in_check {
            return quiescence(ply, alpha, beta, pv, td, board);
        }

        return board.evaluate();
    }

    if ply > 0 {
        if board.is_draw() {
            return STALEMATE;
        }
        // Determines if there is a faster path to checkmate than evaluating the current node, and
        // if there is, it returns early
        let alpha = alpha.max(-CHECKMATE + ply);
        let beta = beta.min(CHECKMATE - ply - 1);
        if alpha >= beta {
            return alpha;
        }
        depth += i32::from(in_check);
    }

    if depth <= 0 {
        return quiescence(ply, alpha, beta, pv, td, board);
    }

    let mut table_move = Move::NULL;
    let entry = td.transpos_table.get(board.zobrist_hash, ply);
    if let Some(entry) = entry {
        let flag = entry.flag();
        let table_eval = entry.search_score();
        table_move = entry.best_move(board);

        if !IS_PV
            && !is_root
            && depth <= entry.depth()
            && match flag {
                EntryFlag::None => false,
                EntryFlag::Exact => true,
                EntryFlag::AlphaUnchanged => table_eval <= alpha,
                EntryFlag::BetaCutOff => table_eval >= beta,
            }
        {
            return table_eval;
        }
    } else if depth >= MIN_IIR_DEPTH && !IS_PV {
        // IIR (Internal Iterative Deepening) - Reduce depth if a node doesn't have a TT hit and isn't a
        // PV node
        depth -= 1;
    }

    let mut best_score = -INFINITY;
    let mut best_move = Move::NULL;
    let original_alpha = alpha;

    let static_eval = if in_check {
        -CHECKMATE
    } else if let Some(entry) = entry {
        entry.static_eval()
    } else {
        board.evaluate()
    };
    td.stack[ply].static_eval = static_eval;
    let improving = !in_check && ply > 1 && static_eval > td.stack[ply - 2].static_eval;

    // TODO: Killers should probably be reset here
    // td.stack[ply].killers = [Move::NULL; 2];

    if !is_root && !IS_PV && !in_check {
        // Reverse futility pruning
        if static_eval - RFP_MULTIPLIER * depth / if improving { 2 } else { 1 } >= beta
            && depth < MAX_RFP_DEPTH
            && static_eval.abs() < NEAR_CHECKMATE
        {
            return static_eval;
        }

        // Null move pruning (NMP)
        if board.has_non_pawns(board.to_move) && depth >= MIN_NMP_DEPTH && static_eval >= beta && board.can_nmp() {
            let mut node_pvs = Vec::new();
            let mut new_b = board.to_owned();
            new_b.make_null_move();
            td.stack[ply].played_move = Move::NULL;
            let r = 3 + depth / 3 + min((static_eval - beta) / 200, 3);
            let mut null_eval = -alpha_beta::<false>(depth - r, -beta, -beta + 1, &mut node_pvs, td, &new_b, !cut_node);
            if null_eval >= beta {
                if null_eval > NEAR_CHECKMATE {
                    null_eval = beta;
                }
                return null_eval;
            }
        }
    }

    let mut moves = board.generate_moves(MGT::All);
    let mut legal_moves_searched = 0;
    moves.score_moves(board, table_move, td.stack[ply].killers, td, ply);

    let mut quiets_tried = Vec::new();
    let mut tacticals_tried = Vec::new();

    // Start of search
    for MoveListEntry { m, score: hist_score } in moves {
        let mut new_b = board.to_owned();
        // let is_quiet = board.is_quiet(m);
        let is_quiet = !m.is_tactical(board);

        if !is_root && best_score >= -NEAR_CHECKMATE {
            if is_quiet {
                // Late move pruning (LMP)
                // By now all quiets have been searched.
                if depth < MAX_LMP_DEPTH
                    && legal_moves_searched > (LMP_CONST + depth * depth) / if improving { 1 } else { 2 }
                {
                    break;
                }
            }
            // TODO: Try -15 * depth * depth for capture
            let margin = if m.is_capture(board) { -90 } else { -50 } * depth;
            if depth < 7 && !board.see(m, margin) {
                continue;
            }
        }

        // Make move filters out illegal moves by returning false if a move was illegal
        if !new_b.make_move::<true>(m) {
            continue;
        }
        if is_quiet {
            quiets_tried.push(m)
        } else {
            tacticals_tried.push(m)
        };

        td.nodes_searched += 1;
        td.stack[ply].played_move = m;
        let mut node_pvs = Vec::new();

        // Calculate the reduction used in LMR
        let r = {
            if legal_moves_searched < LMR_THRESHOLD || depth < MIN_LMR_DEPTH {
                1
            } else {
                let mut r = get_reduction(depth, legal_moves_searched);
                r += i32::from(!IS_PV);
                r += i32::from(!improving);
                if is_quiet && cut_node {
                    r += 2;
                }
                if is_quiet {
                    if hist_score > MAX_HIST_VAL / 2 {
                        r -= 1;
                    } else if hist_score < -MAX_HIST_VAL / 2 {
                        r += 1;
                    }
                }
                if m.is_capture(board) && hist_score < BAD_CAPTURE + 100 {
                    r += 1;
                }
                // Don't let LMR send us into qsearch
                r.clamp(1, depth - 1)
            }
        };

        let eval = if legal_moves_searched == 0 {
            node_pvs.clear();
            // On the first move, just do a full depth search
            -alpha_beta::<IS_PV>(depth - 1, -beta, -alpha, &mut node_pvs, td, &new_b, false)
        } else {
            node_pvs.clear();
            // Start with a zero window reduced search
            let zero_window_reduced_depth =
                -alpha_beta::<false>(depth - r, -alpha - 1, -alpha, &mut node_pvs, td, &new_b, !cut_node);

            // If that search raises alpha and the reduction was more than one, do a research at a zero window with full depth
            let zero_window_full_depth = if zero_window_reduced_depth > alpha && r > 1 {
                node_pvs.clear();
                -alpha_beta::<false>(depth - 1, -alpha - 1, -alpha, &mut node_pvs, td, &new_b, !cut_node)
            } else {
                zero_window_reduced_depth
            };

            // If the verification score falls between alpha and beta, full window full depth search
            if zero_window_full_depth > alpha && zero_window_full_depth < beta {
                node_pvs.clear();
                -alpha_beta::<IS_PV>(depth - 1, -beta, -alpha, &mut node_pvs, td, &new_b, false)
            } else {
                zero_window_full_depth
            }
        };

        legal_moves_searched += 1;

        if eval > best_score {
            best_score = eval;

            if eval > alpha {
                alpha = eval;
                best_move = m;
                store_pv(pv, &mut node_pvs, m);
            }

            if alpha >= beta {
                if is_quiet {
                    // We don't want to store tactical moves, because they are obviously already
                    // good.
                    // Also don't store killers that we have already stored
                    if td.stack[ply].killers[0] != m {
                        td.stack[ply].killers[1] = td.stack[ply].killers[0];
                        td.stack[ply].killers[0] = m;
                    }
                }
                td.history
                    .update_histories(m, &quiets_tried, &tacticals_tried, board, depth, &td.stack, ply);
                break;
            }
        }
    }

    if legal_moves_searched == 0 {
        if board.in_check {
            // Distance from root is returned in order for other recursive calls to determine
            // shortest viable checkmate path
            return -CHECKMATE + ply;
        }
        return STALEMATE;
    }

    let entry_flag = if best_score >= beta {
        EntryFlag::BetaCutOff
    } else if best_score > original_alpha {
        EntryFlag::Exact
    } else {
        EntryFlag::AlphaUnchanged
    };

    td.transpos_table
        .store(board.zobrist_hash, best_move, depth, entry_flag, best_score, ply, IS_PV, static_eval);

    best_score
}

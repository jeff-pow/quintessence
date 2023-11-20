use std::cmp::{max, min};
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::board::board::Board;
use crate::engine::transposition::EntryFlag;
use crate::moves::movegenerator::{generate_moves, MGT};
use crate::moves::movelist::{MoveListEntry, BAD_CAPTURE};
use crate::moves::moves::Move;
use crate::search::{ThreadData, INIT_ASP};

use super::history_heuristics::MAX_HIST_VAL;
use super::quiescence::quiescence;
use super::{
    get_reduction, store_pv, SearchInfo, SearchType, LMP_CONST, LMR_THRESHOLD, MAX_LMP_DEPTH, MAX_RFP_DEPTH,
    MIN_IIR_DEPTH, MIN_LMR_DEPTH, MIN_NMP_DEPTH, RFP_MULTIPLIER,
};

pub const CHECKMATE: i32 = 30000;
pub const STALEMATE: i32 = 0;
pub const NEAR_CHECKMATE: i32 = CHECKMATE - 1000;
pub const INFINITY: i32 = 50000;
pub const MAX_SEARCH_DEPTH: i32 = 100;

pub fn print_search_stats(info: &ThreadData, eval: i32, pv: &[Move]) {
    print!(
        "info time {} seldepth {} depth {} nodes {} nps {} score cp {} pv ",
        info.search_stats.start.elapsed().as_millis(),
        info.sel_depth,
        info.iter_max_depth,
        info.search_stats.nodes_searched,
        (info.search_stats.nodes_searched as f64 / info.search_stats.start.elapsed().as_secs_f64()) as i64,
        eval
    );
    for m in pv {
        print!("{} ", m.to_san());
    }
    println!();
}

pub fn search(info: &mut SearchInfo, mut max_depth: i32) -> Move {
    let mut best_move = Move::NULL;
    let mut pv_moves = Vec::new();

    match info.search_type {
        SearchType::Time => {
            info.game_time.recommended_time(info.board.to_move);
        }
        SearchType::Depth => {
            info.max_depth = max_depth;
        }
        SearchType::Infinite => {
            max_depth = MAX_SEARCH_DEPTH;
            info.max_depth = max_depth;
        }
    }
    info.search_stats.start = Instant::now();

    let mut td = ThreadData::new(&info.transpos_table, &info.halt, &info.game_time, info.board.to_move);

    let mut alpha;
    let mut beta;
    // The previous eval from this side (two moves ago) is a good place to estimate the next
    // aspiration window around. First depth will not have an estimate, and we will do a full
    // window search
    let mut score_history = vec![info.board.evaluate()];
    td.iter_max_depth = 1;

    while td.iter_max_depth <= max_depth {
        td.sel_depth = 0;
        let board = &info.board.to_owned();

        // We assume the average eval for the board from two iterations ago is a good estimate for
        // the next iteration
        let prev_avg = if td.iter_max_depth >= 2 {
            *score_history.get(td.iter_max_depth as usize - 2).unwrap() as f64
        } else {
            -INFINITY as f64
        };

        // Create a window we think the actual evaluation of the position will fall within
        let mut delta = INIT_ASP + (prev_avg * prev_avg * 6.25e-5) as i32;
        alpha = max(prev_avg as i32 - delta, -INFINITY);
        beta = min(prev_avg as i32 + delta, INFINITY);

        let mut score;
        loop {
            score = alpha_beta::<true>(td.iter_max_depth, alpha, beta, &mut pv_moves, &mut td, board, false);
            if score <= alpha {
                beta = (alpha + beta) / 2;
                alpha = max(score - delta, -INFINITY);
            } else if score >= beta {
                beta = min(score + delta, INFINITY);
            } else {
                break;
            }
            delta += delta / 3;
            debug_assert!(alpha >= -INFINITY && beta <= INFINITY);
        }

        if !pv_moves.is_empty() {
            best_move = pv_moves[0];
        }
        score_history.push(score);

        print_search_stats(&td, score, &pv_moves);

        if info.search_type == SearchType::Time && info.game_time.soft_termination(info.search_stats.start) {
            break;
        }
        if info.halt.load(Ordering::SeqCst) {
            break;
        }
        td.iter_max_depth += 1;
    }

    info.search_stats.nodes_searched = td.search_stats.nodes_searched;
    assert_ne!(best_move, Move::NULL);

    best_move
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
    if td.search_stats.nodes_searched % 1024 == 0
        && (td.halt.load(Ordering::Relaxed) || td.game_time.hard_termination(td.search_stats.start))
    {
        return board.evaluate();
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
        let table_eval = entry.eval();
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
    td.stack[ply as usize].static_eval = static_eval;
    let improving = !in_check && ply > 1 && static_eval > td.stack[ply as usize - 2].static_eval;

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

    let mut moves = generate_moves(board, MGT::All);
    let mut legal_moves_searched = 0;
    moves.score_moves(board, table_move, td.stack[ply as usize].killers, td);
    td.search_stats.nodes_searched += 1;

    // Start of search
    for MoveListEntry { m, score: hist_score } in moves {
        let mut new_b = board.to_owned();
        let is_quiet = board.is_quiet(m);

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
        if !new_b.make_move(m) {
            continue;
        }
        td.current_line.push(m);
        td.stack[ply as usize].played_move = m;
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
        td.current_line.pop();

        if eval > best_score {
            best_score = eval;

            if eval > alpha {
                alpha = eval;
                best_move = m;
                store_pv(pv, &mut node_pvs, m);
            }

            if alpha >= beta {
                if let Some(cap) = board.capture(m) {
                    td.history.update_capt_hist(m, board.to_move, cap, depth, true);
                } else {
                    // Store a killer move if it is not a capture, but good enough to cause a beta cutoff
                    // Also don't store killers that we have already stored
                    let ply = ply as usize;
                    // Store killer move
                    if td.stack[ply].killers[0] != m {
                        td.stack[ply].killers[1] = td.stack[ply].killers[0];
                        td.stack[ply].killers[0] = m;
                    }
                    td.history.update_quiet_history(m, true, board.to_move, depth);
                    td.history
                        .set_counter(board.to_move, *td.current_line.last().unwrap_or(&Move::NULL), m);
                }
                break;
            }
        }

        // If a move doesn't raise alpha, deduct from its history score for move ordering
        if board.is_quiet(m) {
            td.history.update_quiet_history(m, false, board.to_move, depth);
        } else {
            td.history
                .update_capt_hist(m, board.to_move, board.capture(m).unwrap(), depth, false);
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

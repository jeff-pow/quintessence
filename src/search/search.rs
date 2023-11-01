use std::cmp::{max, min};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::board::board::Board;
use crate::engine::transposition::{EntryFlag, TableEntry};
use crate::eval::eval::evaluate;
use crate::moves::movegenerator::{generate_moves, MGT};
use crate::moves::movelist::MoveListEntry;
use crate::moves::moves::Move;

use super::history_heuristics::MAX_HIST_VAL;
use super::killers::store_killer_move;
use super::quiescence::quiescence;
use super::see::see;
use super::{get_reduction, store_pv, SearchInfo, SearchType};

pub const CHECKMATE: i32 = 30000;
pub const STALEMATE: i32 = 0;
pub const NEAR_CHECKMATE: i32 = CHECKMATE - 1000;
pub const INFINITY: i32 = 50000;
pub const MAX_SEARCH_DEPTH: i32 = 100;

// Tunable Constants
/// Initial aspiration window value
pub const INIT_ASP: i32 = 10;

/// Begin LMR if more than this many moves have been searched
// TODO: Test this at 1
pub const LMR_THRESHOLD: i32 = 2;
pub const MIN_LMR_DEPTH: i32 = 2;
const MAX_CAPTURE_SEE_DEPTH: i32 = 6;
const CAPTURE_SEE_COEFFICIENT: i32 = 15;
const MAX_QUIET_SEE_DEPTH: i32 = 8;
const QUIET_SEE_COEFFICIENT: i32 = 50;
const MAX_LMP_DEPTH: i32 = 6;
const LMP_DIVISOR: i32 = 2;
const LMP_CONST: i32 = 3;
const RFP_MULTIPLIER: i32 = 70;
const MAX_RFP_DEPTH: i32 = 9;
const MIN_NMP_DEPTH: i32 = 3;
const MIN_IIR_DEPTH: i32 = 4;

pub fn print_search_stats(info: &SearchInfo, eval: i32, pv: &[Move], iter_depth: i32) {
    print!(
        "info time {} seldepth {} depth {} nodes {} nps {} score cp {} pv ",
        info.search_stats.start.elapsed().as_millis(),
        info.sel_depth,
        iter_depth,
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

    let mut recommended_time = Duration::ZERO;
    match info.search_type {
        SearchType::Time => {
            recommended_time = info.game_time.recommended_time(info.board.to_move);
        }
        SearchType::Depth => (),
        SearchType::Infinite => {
            max_depth = MAX_SEARCH_DEPTH;
            info.max_depth = max_depth;
        }
    }
    info.search_stats.start = Instant::now();

    let mut alpha;
    let mut beta;
    // The previous eval from this side (two moves ago) is a good place to estimate the next
    // aspiration window around. First depth will not have an estimate, and we will do a full
    // window search
    let mut score_history = vec![evaluate(&info.board)];
    info.iter_max_depth = 1;

    while info.iter_max_depth <= max_depth {
        info.sel_depth = 0;
        let board = &info.board.to_owned();

        // We assume the average eval for the board from two iterations ago is a good estimate for
        // the next iteration
        let prev_avg = if info.iter_max_depth >= 2 {
            *score_history.get(info.iter_max_depth as usize - 2).unwrap() as f64
        } else {
            -INFINITY as f64
        };

        // Create a window we think the actual evaluation of the position will fall within
        let mut delta = INIT_ASP + (prev_avg * prev_avg * 6.25e-5) as i32;
        alpha = max(prev_avg as i32 - delta, -INFINITY);
        beta = min(prev_avg as i32 + delta, INFINITY);

        let mut score;
        loop {
            score = alpha_beta(info.iter_max_depth, alpha, beta, &mut pv_moves, info, board, false);
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

        print_search_stats(info, score, &pv_moves, info.iter_max_depth);

        if info.search_type == SearchType::Time
            && info
                .game_time
                .reached_termination(info.search_stats.start, recommended_time)
        {
            break;
        }
        if info.halt.load(Ordering::SeqCst) {
            break;
        }
        info.iter_max_depth += 1;
    }

    assert_ne!(best_move, Move::NULL);

    best_move
}

/// Principal variation search - uses reduced alpha beta windows around a likely best move candidate
/// to refute other variations
#[allow(clippy::too_many_arguments)]
fn alpha_beta(
    mut depth: i32,
    mut alpha: i32,
    beta: i32,
    pv: &mut Vec<Move>,
    info: &mut SearchInfo,
    board: &Board,
    cut_node: bool,
) -> i32 {
    let ply = info.iter_max_depth - depth;
    let is_root = ply == 0;
    let in_check = board.in_check(board.to_move);
    // Is a zero width search if alpha and beta are one apart
    let is_pv_node = (beta - alpha).abs() != 1;
    info.sel_depth = info.sel_depth.max(ply);
    // Don't do pvs unless you have a pv - otherwise you're wasting time
    if info.halt.load(Ordering::SeqCst) {
        return evaluate(board);
    }

    // if in_check {
    //     depth += 1;
    // }

    // Needed since the function can calculate extensions in cases where it finds itself in check
    if ply >= MAX_SEARCH_DEPTH {
        if board.in_check(board.to_move) {
            return quiescence(ply, alpha, beta, pv, info, board);
        }
        return evaluate(board);
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
    }

    // Attempt to read eval, or at least a suggestion for the best move from transposition table
    let (table_value, table_move) = {
        if let Some(entry) = info.transpos_table.read().unwrap().get(&board.zobrist_hash) {
            entry.get(depth, ply, alpha, beta, board)
        } else {
            (None, Move::NULL)
        }
    };
    if let Some(eval) = table_value {
        if !is_root && !is_pv_node {
            // This can cut off evals in certain cases, but it's easy to implement :)
            return eval;
        }
    }
    // IIR (Internal Iterative Deepening) - Reduce depth if a node doesn't have a TT eval, isn't a
    // PV node, and is a cutNode (?)
    else if depth >= MIN_IIR_DEPTH && !is_pv_node {
        depth -= 1;
    }

    if depth <= 0 {
        return quiescence(ply, alpha, beta, pv, info, board);
    }

    let mut best_score = -INFINITY;
    let mut best_move = Move::NULL;
    let original_alpha = alpha;
    let hist_bonus = (155 * depth).min(2000);

    if !is_root && !is_pv_node && !in_check {
        let static_eval = evaluate(board);
        // Reverse futility pruning
        if static_eval - RFP_MULTIPLIER * depth >= beta && depth < MAX_RFP_DEPTH && static_eval.abs() < NEAR_CHECKMATE {
            return static_eval;
        }

        // Null move pruning (NMP)
        if board.has_non_pawns(board.to_move) && depth >= MIN_NMP_DEPTH && static_eval >= beta && board.can_nmp() {
            let mut node_pvs = Vec::new();
            let mut new_b = board.to_owned();
            new_b.to_move = !new_b.to_move;
            new_b.en_passant_square = None;
            new_b.prev_move = Move::NULL;
            let r = 3 + depth / 3 + min((static_eval - beta) / 200, 3);
            let mut null_eval = -alpha_beta(depth - r, -beta, -beta + 1, &mut node_pvs, info, &new_b, !cut_node);
            if null_eval >= beta {
                if null_eval > NEAR_CHECKMATE {
                    null_eval = beta;
                }
                if info.nmp_plies == 0 || depth < 10 {
                    return null_eval;
                }
                // Concept from stockfish
                info.nmp_plies = ply + (depth - r) / 3;
                let null_eval = alpha_beta(depth - r, beta - 1, beta, &mut Vec::new(), info, board, false);
                info.nmp_plies = 0;
                if null_eval >= beta {
                    return null_eval;
                }
            }
        }
    }

    let mut moves = generate_moves(board, MGT::All);
    let mut legal_moves_searched = 0;
    moves.score_moves(board, table_move, &info.killer_moves[ply as usize], info);
    info.search_stats.nodes_searched += 1;

    // Start of search
    for MoveListEntry { m, score: hist_score } in moves {
        let mut new_b = board.to_owned();
        let is_quiet = board.is_quiet(m);

        if !is_root && best_score >= -NEAR_CHECKMATE {
            if is_quiet {
                // Late move pruning (LMP)
                // By now all quiets have been searched.
                if depth < MAX_LMP_DEPTH
                    && !is_pv_node
                    && !in_check
                    && legal_moves_searched > (LMP_CONST + depth * depth) / LMP_DIVISOR
                {
                    break;
                }

                // Quiet SEE pruning
                if depth <= MAX_QUIET_SEE_DEPTH && !see(board, m, -QUIET_SEE_COEFFICIENT * depth) {
                    continue;
                }
            } else {
                // Capture SEE pruning
                if depth <= MAX_CAPTURE_SEE_DEPTH && !see(board, m, -CAPTURE_SEE_COEFFICIENT * depth * depth) {
                    continue;
                }
            }
        }

        // Make move filters out illegal moves
        if !new_b.make_move(m) {
            continue;
        }
        info.current_line.push(m);
        let mut node_pvs = Vec::new();

        let can_lmr = legal_moves_searched >= LMR_THRESHOLD && depth >= MIN_LMR_DEPTH;
        // R is used in late move reductions
        let r = {
            if !can_lmr {
                1
            } else {
                let mut r = get_reduction(info, depth, legal_moves_searched);
                r += i32::from(!is_pv_node);
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
                // if is_quiet && !see(&new_b, m, -50 * depth) {
                //     depth += 1;
                // }
                // Don't let LMR send us into qsearch
                r.clamp(1, depth - 1)
            }
        };

        let eval = if legal_moves_searched == 0 {
            node_pvs.clear();
            // On the first move, just do a full depth search so we at least have a pv
            -alpha_beta(depth - 1, -beta, -alpha, &mut node_pvs, info, &new_b, false)
        } else {
            node_pvs.clear();
            // Start with a zero window reduced search
            let zero_window = -alpha_beta(depth - r, -alpha - 1, -alpha, &mut Vec::new(), info, &new_b, !cut_node);

            // If that search raises alpha and the reduction was more than one, do a research at a zero window with full depth
            let verification_score = if zero_window > alpha && r > 1 {
                node_pvs.clear();
                -alpha_beta(depth - 1, -alpha - 1, -alpha, &mut node_pvs, info, &new_b, !cut_node)
            } else {
                zero_window
            };

            // If the verification score falls between alpha and beta, full window full depth search
            if verification_score > alpha && verification_score < beta {
                node_pvs.clear();
                -alpha_beta(depth - 1, -beta, -alpha, &mut node_pvs, info, &new_b, false)
            } else {
                verification_score
            }
        };

        legal_moves_searched += 1;
        info.current_line.pop();

        if eval > best_score {
            best_score = eval;
            best_move = m;
            if eval > alpha {
                alpha = eval;
                best_move = m;
                store_pv(pv, &mut node_pvs, m);
            }
            if alpha >= beta {
                let capture = board.capture(m);
                // Store a killer move if it is not a capture, but good enough to cause a beta cutoff
                // Also don't store killers that we have already stored
                if capture.is_none() {
                    store_killer_move(ply, m, info);
                    info.history.update_history(m, hist_bonus, board.to_move);
                    info.history
                        .set_counter(board.to_move, *info.current_line.last().unwrap_or(&Move::NULL), m);
                }
                break;
            }
        }
        // TODO: Try only doing this for quiet moves
        // If a move doesn't raise alpha, deduct from its history score for move ordering
        info.history.update_history(m, -hist_bonus, board.to_move);
    }

    if legal_moves_searched == 0 {
        if board.in_check(board.to_move) {
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

    info.transpos_table
        .write()
        .unwrap()
        .insert(board.zobrist_hash, TableEntry::new(depth, ply, entry_flag, best_score, best_move));

    best_score
}

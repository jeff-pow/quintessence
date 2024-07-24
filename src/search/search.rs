use std::cmp::{max, min};
use std::time::Instant;

use crate::board::Board;
use crate::chess_move::Move;
use crate::movelist::{MoveListEntry, MAX_LEN};
use crate::movepicker::MovePicker;
use crate::search::SearchStack;
use crate::thread::ThreadData;
use crate::transposition::{EntryFlag, TableEntry, TranspositionTable};

use super::quiescence::quiescence;
use super::PV;
use arrayvec::ArrayVec;

pub const CHECKMATE: i32 = 25000;
pub const STALEMATE: i32 = 0;
pub const NEAR_CHECKMATE: i32 = CHECKMATE - 1000;
pub const INFINITY: i32 = 30000;
pub const MAX_SEARCH_DEPTH: i32 = 100;

pub fn start_search(td: &mut ThreadData, print_uci: bool, board: Board, tt: &TranspositionTable) {
    td.search_start = Instant::now();
    td.nodes_table = [[0; 64]; 64];
    td.best_move = Move::NULL;
    td.stack = SearchStack::default();
    td.accumulators.clear(&board.new_accumulator());

    iterative_deepening(td, &board, print_uci, tt);
}

/// Rather than sticking to a fixed depth for search, gradually ramping up the search depth by one
/// level until time expires actually saves time. This method relies on earlier depth searches
/// finishing quickly, building up important structures like transposition and history tables along
/// the way. As a result, for more expensive depths, we already have a good idea of the best move
/// and can maximize the efficacy of alpha beta pruning.
pub fn iterative_deepening(td: &mut ThreadData, board: &Board, print_uci: bool, tt: &TranspositionTable) {
    let mut pv = PV::default();
    let mut prev_score = -INFINITY;
    let mut depth = 1;

    loop {
        td.sel_depth = 0;

        assert_eq!(0, td.ply);
        assert_eq!(0, td.accumulators.top);

        prev_score = aspiration_windows(td, &mut pv, prev_score, board, tt, depth);

        assert_eq!(0, td.accumulators.top);

        if td.halt() {
            break;
        }

        // Only update best move if the search wasn't aborted
        td.best_move = pv.line[0];

        if print_uci {
            td.print_search_stats(prev_score, &pv, tt, depth);
        }

        if td.soft_stop(depth, prev_score) {
            td.set_halt(true);
            break;
        }

        depth += 1;
    }

    assert_ne!(td.best_move, Move::NULL);
}

/// Aspiration windows place a bound around the likely range the score for a search will fall
/// within which means we run into cutoffs if the score exceeds either side of the range we
/// predicted, leading to a faster search than a full alpha-beta window each search.
fn aspiration_windows(
    td: &mut ThreadData,
    pv: &mut PV,
    prev_score: i32,
    board: &Board,
    tt: &TranspositionTable,
    iter_depth: i32,
) -> i32 {
    let mut alpha = -INFINITY;
    let mut beta = INFINITY;
    // Asp window should start wider if score is more extreme
    let mut delta = 5 + prev_score * prev_score / 9534;

    let mut depth = iter_depth;

    if depth >= 2 {
        alpha = alpha.max(prev_score - delta);
        beta = beta.min(prev_score + delta);
    }

    loop {
        assert_eq!(0, td.ply);
        let score = negamax::<true>(depth, alpha, beta, pv, td, tt, board, false);

        if td.halt() {
            return score;
        }

        if score <= alpha {
            beta = (alpha + beta) / 2;
            alpha = max(score - delta, -INFINITY);
            // If move/position proves to not be as good as we thought, we need to do a full depth
            // search to get the best idea possible of its actual score.
            depth = iter_depth;
        } else if score >= beta {
            beta = min(score + delta, INFINITY);
            // If window is better than beta, we have a potentially untrustworthy best move that we
            // want to prove is safe quickly, so we reduce depth.
            depth -= 1;
        } else {
            return score;
        }
        delta += 4 * delta / 9;
    }
}

/// Principal variation search - uses reduced alpha beta windows around a likely best move candidate
/// to refute other variations
///
/// cut_node is a parameter that predicts whether or not a node will fail high or not. If cut_node
/// is true, we expect a beta cutoff or fail high to occur.
///
/// IS_PV denotes a node's PV status. PV nodes (generally) have a difference between alpha and beta
/// of > 1, while in non-PV nodes the window is always beta - alpha = 1. Once a node loses its PV
/// status, it can never regain it, so the majority of nodes searched are non-PV.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn negamax<const IS_PV: bool>(
    mut depth: i32,
    mut alpha: i32,
    beta: i32,
    pv: &mut PV,
    td: &mut ThreadData,
    tt: &TranspositionTable,
    board: &Board,
    cut_node: bool,
) -> i32 {
    let is_root = td.ply == 0;
    let in_check = board.in_check();

    let singular_move = td.stack[td.ply].singular;
    let singular_search = singular_move != Move::NULL;

    td.sel_depth = td.sel_depth.max(td.ply);

    if td.main_thread() && td.hard_stop() {
        td.set_halt(true);
        return 0;
    }

    if depth <= 0 && !in_check {
        return quiescence::<IS_PV>(alpha, beta, pv, td, tt, board);
    }

    depth = depth.max(0);

    // Don't prune at root to ensure we have a best move
    if !is_root {
        if td.halt() {
            return 0;
        }

        if board.is_draw() || td.is_repetition(board) {
            return STALEMATE;
        }

        if td.ply >= MAX_SEARCH_DEPTH - 1 {
            return if in_check { 0 } else { td.accumulators.evaluate(board) };
        }

        // Determines if there is a faster path to checkmate than evaluating the current node, and
        // if there is, it returns early
        let alpha = alpha.max(-CHECKMATE + td.ply);
        let beta = beta.min(CHECKMATE - td.ply - 1);
        if alpha >= beta {
            return alpha;
        }
    }

    // Attempt to look up information from previous searches in the same board state
    let mut tt_move = Move::NULL;
    let mut tt_pv = IS_PV;
    let entry = tt.get(board.zobrist_hash, td.ply);
    if let Some(entry) = entry {
        let tt_flag = entry.flag();
        let tt_score = entry.search_score();
        tt_move = entry.best_move(board);
        tt_pv |= entry.was_pv();

        // Don't do TT cutoffs in verification search for singular moves
        if !singular_search
            && !IS_PV
            && depth <= entry.depth()
            && match tt_flag {
                EntryFlag::None => false,
                EntryFlag::Exact => true,
                EntryFlag::AlphaUnchanged => tt_score <= alpha,
                EntryFlag::BetaCutOff => tt_score >= beta,
            }
        {
            return tt_score;
        }
    } else if depth >= 1 && !IS_PV && !singular_search {
        // IIR (Internal Iterative Deepening) - Reduce depth if a node doesn't have a TT hit and isn't a
        // PV node
        // TODO: Unlink IIR from the entry existing - just check if tt move is null instead
        // TODO: Try high depth cutnode iir as seen in SF
        depth -= 1;
    }

    let mut best_score = -INFINITY;
    let mut best_move = Move::NULL;
    let original_alpha = alpha;

    let raw_eval;
    let corrected_eval;
    let estimated_eval;
    if board.in_check() {
        raw_eval = -INFINITY;
        corrected_eval = -INFINITY;
        estimated_eval = -INFINITY;
    } else if let Some(entry) = entry {
        // Get static eval from transposition table if possible
        raw_eval = if entry.static_eval() != -INFINITY { entry.static_eval() } else { td.accumulators.evaluate(board) };
        corrected_eval = td.history.corr_hist.correct_score(board.stm, board.pawn_hash, raw_eval);
        if entry.search_score() != -INFINITY
            && (entry.flag() == EntryFlag::AlphaUnchanged && entry.search_score() < raw_eval
                || entry.flag() == EntryFlag::BetaCutOff && entry.search_score() > raw_eval
                || entry.flag() == EntryFlag::Exact)
        {
            estimated_eval = entry.search_score();
        } else {
            estimated_eval = corrected_eval;
        }
    } else {
        raw_eval = td.accumulators.evaluate(board);
        tt.store(board.zobrist_hash, Move::NULL, 0, EntryFlag::None, -INFINITY, td.ply, tt_pv, raw_eval);
        corrected_eval = td.history.corr_hist.correct_score(board.stm, board.pawn_hash, raw_eval);
        estimated_eval = corrected_eval;
    };

    td.stack[td.ply].static_eval = estimated_eval;
    let improving = {
        if in_check {
            false
        } else if td.ply > 1 && td.stack[td.ply - 2].static_eval != -INFINITY {
            estimated_eval > td.stack[td.ply - 2].static_eval
        } else if td.ply > 3 && td.stack[td.ply - 4].static_eval != -INFINITY {
            estimated_eval > td.stack[td.ply - 4].static_eval
        } else {
            // This could be true or false, could experiment with it in the future
            false
        }
    };

    // TODO: Killers should probably be reset here
    // td.stack[td.ply + 1].killers = [Move::NULL; 2];
    td.stack[td.ply + 1].singular = Move::NULL;
    td.stack[td.ply + 2].cutoffs = 0;
    if !is_root {
        td.stack[td.ply].multi_extns = td.stack[td.ply - 1].multi_extns;
    }

    // Pre-move loop pruning
    let can_prune = !IS_PV && !in_check && !singular_search;

    // Reverse futility pruning (RFP) - If we are below beta by a certain amount, we are unlikely to
    // raise it, so we can prune the nodes that would have followed
    if can_prune
        && estimated_eval - 93 * depth + i32::from(improving) * 30 * depth >= beta
        && estimated_eval >= beta
        && depth < 9
        && estimated_eval.abs() < NEAR_CHECKMATE
    {
        // TODO: Make sure this returns a score < checkmate
        return (estimated_eval + beta) / 2;
    }

    // Null move pruning (NMP) - If we can give the opponent a free move and they still can't
    // raise beta at a reduced depth search, they likely won't be able to if we move either,
    // so we can prune the nodes that would have followed
    if can_prune
        && board.has_non_pawns(board.stm)
        && depth >= 2
        && estimated_eval >= beta
        && td.stack[td.ply - 1].played_move != Move::NULL
        && beta > -NEAR_CHECKMATE
    {
        let mut node_pv = PV::default();
        let mut new_b = *board;

        tt.prefetch(board.hash_after(Move::NULL));
        new_b.make_null_move();
        td.stack[td.ply].played_move = Move::NULL;
        td.hash_history.push(new_b.zobrist_hash);
        td.ply += 1;

        // Reduction
        let r = 4 + depth / 4 + min((estimated_eval - beta) / 173, 7);
        let mut null_eval = -negamax::<false>(depth - r, -beta, -beta + 1, &mut node_pv, td, tt, &new_b, !cut_node);

        td.hash_history.pop();
        td.ply -= 1;
        if td.halt() {
            return 0;
        }
        // TODO: NMP verification search

        if null_eval >= beta {
            // Ensure we don't return a checkmate score
            if null_eval > NEAR_CHECKMATE {
                null_eval = beta;
            }
            return null_eval;
        }
    }

    let mut moves_searched = 0;
    let mut picker = MovePicker::new(tt_move, td, -197, false);

    let mut quiets_tried = ArrayVec::<_, MAX_LEN>::new();
    let mut tacticals_tried = ArrayVec::<_, MAX_LEN>::new();

    // Start of search
    while let Some(MoveListEntry { m, .. }) = picker.next(board, td) {
        // Don't consider the singular move in a verification search
        if Some(m) == singular_move {
            continue;
        }
        let is_quiet = !m.is_tactical(board);

        // Mid-move loop pruning
        if !is_root && best_score >= -NEAR_CHECKMATE {
            if is_quiet {
                // Late move pruning (LMP)
                // Good moves are likely to be searched first due to tt move ordering and history
                // table, so we can prune all the moves that follow as they are very unlikely to be good.
                let moves_required = if improving {
                    2.48 + 0.5 * depth as f32 * depth as f32
                } else {
                    1.08 + 0.26 * depth as f32 * depth as f32
                } as i32;
                // TODO: Remove depth condition
                if depth < 11 && moves_searched > moves_required {
                    break;
                }

                // Futility pruning
                let lmr_depth = (depth - td.lmr.base_reduction(depth, moves_searched)).max(0);
                if !singular_search
                    && !in_check
                    && lmr_depth < 11
                    && estimated_eval + 199 + 69 * lmr_depth <= alpha
                    && alpha < NEAR_CHECKMATE
                {
                    break;
                }
            }

            // Static exchange pruning - If we fail to immediately recapture a depth dependent
            // threshold, don't bother searching the move
            let margin = if m.is_capture(board) { -93 } else { -41 } * depth;
            if depth < 12 && !board.see(m, margin) {
                continue;
            }
        }

        let mut new_b = *board;
        tt.prefetch(board.hash_after(Some(m)));
        // Make move filters out illegal moves by returning false if a move was illegal
        if !new_b.make_move(m) {
            continue;
        }
        td.accumulators.push_move(m, board.piece_at(m.to()));

        if is_quiet {
            quiets_tried.push(m);
        } else {
            tacticals_tried.push(m);
        };

        // Extensions are the counterpart to late move reductions. We want to explore promising
        // moves more fully, though in some conditions we also reduce the depth to search at via
        // negative extensions
        let extension = extension::<IS_PV>(entry, alpha, beta, m, depth, board, td, tt, cut_node);

        let mut new_depth = depth + extension - 1;

        td.nodes.increment();
        let pre_search_nodes = td.nodes.local_count();
        td.stack[td.ply].played_move = Some(m);
        td.hash_history.push(new_b.zobrist_hash);
        td.ply += 1;
        let mut node_pv = PV::default();
        let mut eval = -INFINITY;
        let history =
            if is_quiet { td.history.quiet_history(m, &td.stack, td.ply) } else { td.history.capt_hist(m, board) };

        // Late Move Reductions (LMR) - Search moves after the first with reduced depth and
        // window as they are much less likely to be the best move than the first move
        // selected by the move picker.
        if depth > 2 && moves_searched > 1 + i32::from(is_root) && (is_quiet || !tt_pv) {
            let mut r = td.lmr.base_reduction(depth, moves_searched);
            if cut_node {
                r += 1 + i32::from(!m.is_tactical(board));
            }
            // This technically looks one ply into the future since ply is incremented a few lines
            // prior.
            r -= i32::from(td.stack[td.ply].cutoffs < 4);

            r += ((alpha - estimated_eval) / 337).clamp(0, 2);

            r -= history / 9698;

            if tt_pv {
                r -= 1 + i32::from(cut_node)
            }

            // Calculate a reduction and calculate a reduced depth, ensuring we won't drop to depth
            // zero and thus straight into qsearch.
            let d = max(1, min(new_depth - r, new_depth + 1));
            // Preform the zero window, reduced depth search
            eval = -negamax::<false>(d, -alpha - 1, -alpha, &mut node_pv, td, tt, &new_b, true);

            // If eval would raise alpha and calculated reduced depth is actually less than our
            // full depth search (including extensions), search again
            if eval > alpha && d < new_depth {
                new_depth += i32::from(eval > best_score + 40);
                new_depth -= i32::from(eval < best_score + new_depth);
                if new_depth > d {
                    eval = -negamax::<false>(new_depth, -alpha - 1, -alpha, &mut node_pv, td, tt, &new_b, !cut_node);
                }
            }
        }
        // If LMR was not performed, conduct a zero window full depth search on the first move of
        // non-PV nodes (which already have a zero window b/t alpha and beta), or the moves
        // following the first move for PV nodes
        else if moves_searched > 0 || !IS_PV {
            eval = -negamax::<false>(new_depth, -alpha - 1, -alpha, &mut node_pv, td, tt, &new_b, !cut_node);
        }

        // If the node is a PV node and the score calculated in a previous search fell between
        // alpha and beta (an exact score) or no moves have been searched from the current
        // position, execute a full window full depth search.
        if IS_PV && (moves_searched == 0 || (eval > alpha && eval < beta)) {
            eval = -negamax::<true>(new_depth, -beta, -alpha, &mut node_pv, td, tt, &new_b, false);
        }

        if is_root {
            td.nodes_table[m.from()][m.to()] += td.nodes.local_count() - pre_search_nodes;
        }
        moves_searched += 1;
        td.hash_history.pop();
        td.accumulators.pop();
        td.ply -= 1;

        if td.halt() {
            return 0;
        }

        best_score = eval.max(best_score);

        if eval <= alpha {
            continue;
        }

        alpha = eval;
        best_move = Some(m);
        if IS_PV {
            pv.update(best_move, node_pv);
        }

        if eval < beta {
            continue;
        }

        // Prefetch here since we're going to want to write to the tt for this board in a
        // few lines anyway. Probably pretty pointless but I assume that history updates
        // will take enough time to do something. Not empirically tested, but fight me :)
        tt.prefetch(board.zobrist_hash);

        td.stack[td.ply].cutoffs += 1;

        if is_quiet {
            // We don't want to store tactical moves in our killer moves, because they are obviously already
            // good.
            td.stack[td.ply].killer_move = Some(m);
        }
        // Update history tables on a beta cutoff
        td.history.update_histories(m, &quiets_tried, &tacticals_tried, board, depth, &td.stack, td.ply);
        break;
    }

    if moves_searched == 0 {
        best_score = if singular_search {
            alpha
        } else if in_check {
            // Distance from root is returned in order for other recursive calls to determine
            // shortest viable checkmate path
            -CHECKMATE + td.ply
        } else {
            STALEMATE
        };
    }

    let entry_flag = if best_score >= beta {
        EntryFlag::BetaCutOff
    } else if best_score > original_alpha {
        EntryFlag::Exact
    } else {
        EntryFlag::AlphaUnchanged
    };

    if !(in_check
        || (best_move.is_some() && best_move.unwrap().is_tactical(board))
        || (entry_flag == EntryFlag::BetaCutOff && best_score <= corrected_eval)
        || entry_flag == EntryFlag::AlphaUnchanged && best_score >= corrected_eval)
    {
        td.history.corr_hist.update_table(board.stm, board.pawn_hash, depth, best_score - corrected_eval);
    }

    // Don't save to TT while in a singular extension verification search
    if !singular_search {
        tt.store(board.zobrist_hash, best_move, depth, entry_flag, best_score, td.ply, tt_pv, raw_eval);
    }

    best_score
}

#[allow(clippy::too_many_arguments)]
fn extension<const IS_PV: bool>(
    tt_entry: Option<TableEntry>,
    alpha: i32,
    beta: i32,
    m: Move,
    depth: i32,
    board: &Board,
    td: &mut ThreadData,
    tt: &TranspositionTable,
    cut_node: bool,
) -> i32 {
    let Some(entry) = tt_entry else { return 0 };
    let tt_move = entry.best_move(board);
    if entry.depth() < depth - 3
        || matches!(entry.flag(), EntryFlag::AlphaUnchanged | EntryFlag::None)
        || Some(m) != tt_move
        || depth < 7
        || td.ply == 0
    {
        return 0;
    }
    let Some(tt_move) = tt_move else { return 0 };

    let ext_beta = (entry.search_score() - 21 * depth / 16).max(-CHECKMATE);
    let ext_depth = (depth - 1) / 2;
    let mut node_pv = PV::default();
    let npv = &mut node_pv;

    td.stack[td.ply].singular = Some(m);
    let prev = td.accumulators.pop();
    let ext_score = negamax::<false>(ext_depth, ext_beta - 1, ext_beta, npv, td, tt, board, cut_node);
    td.stack[td.ply].singular = Move::NULL;
    td.accumulators.push(prev);

    if ext_score < ext_beta {
        if td.stack[td.ply].multi_extns < 10 && !IS_PV && ext_score < ext_beta - 18 {
            td.stack[td.ply].multi_extns += 1;
            2 + i32::from(!tt_move.is_tactical(board) && ext_score < ext_beta - 224)
        } else {
            1
        }
    } else if entry.search_score() >= beta {
        -2 + i32::from(IS_PV)
    } else if cut_node {
        -2
    } else if entry.search_score() <= alpha {
        -1
    } else {
        0
    }
}

use std::cmp::{max, min};
use std::time::{Duration, Instant};

use crate::board::board::Board;
use crate::engine::transposition::{EntryFlag, TableEntry};
use crate::moves::movegenerator::generate_psuedolegal_moves;
use crate::moves::movelist::{score_move_list, sort_next_move};
use crate::moves::moves::Move;
use crate::types::pieces::{value, PieceName, BISHOP_PTS, QUEEN_PTS, ROOK_PTS};

use super::eval::eval;
use super::killers::store_killer_move;
use super::quiescence::quiescence;
use super::{SearchInfo, SearchType};

pub const CHECKMATE: i32 = 100000;
pub const STALEMATE: i32 = 0;
pub const NEAR_CHECKMATE: i32 = CHECKMATE - 1000;
pub const INFINITY: i32 = 9999999;
pub const MAX_SEARCH_DEPTH: i8 = 100;

pub fn search(search_info: &mut SearchInfo) -> Move {
    let max_depth;
    let mut best_move = Move::NULL;
    let mut pv_moves = Vec::new();

    let mut recommended_time = Duration::ZERO;
    match search_info.search_type {
        SearchType::Time => {
            recommended_time = search_info
                .game_time
                .recommended_time(search_info.board.to_move);
            max_depth = MAX_SEARCH_DEPTH;
        }
        SearchType::Depth => {
            max_depth = search_info.iter_max_depth;
        }
        SearchType::Infinite => {
            search_info.iter_max_depth = MAX_SEARCH_DEPTH;
            max_depth = MAX_SEARCH_DEPTH;
        }
    }

    search_info.search_stats.start = Instant::now();
    let mut iter_depth = 1;

    while iter_depth <= max_depth {
        search_info.iter_max_depth = iter_depth;
        search_info.sel_depth = iter_depth;

        let board = &search_info.board.to_owned();
        let eval = pvs(
            iter_depth,
            -INFINITY,
            INFINITY,
            &mut pv_moves,
            search_info,
            board,
        );

        if !pv_moves.is_empty() {
            best_move = pv_moves[0];
        }
        print!(
            "info time {} seldepth {} depth {} nodes {} nps {} score cp {} pv ",
            search_info.search_stats.start.elapsed().as_millis(),
            search_info.sel_depth,
            iter_depth,
            search_info.search_stats.nodes_searched,
            search_info.search_stats.nodes_searched as f64
                / search_info.search_stats.start.elapsed().as_secs_f64(),
            eval
        );
        for m in pv_moves.iter() {
            print!("{} ", m.to_lan());
        }
        println!();
        if search_info.search_type == SearchType::Time
            && search_info
                .game_time
                .reached_termination(search_info.search_stats.start, recommended_time)
        {
            break;
        }
        iter_depth += 1;
    }

    assert_ne!(best_move, Move::NULL);

    best_move
}

const FUTIL_MARGIN: i32 = BISHOP_PTS;
const FUTIL_DEPTH: i8 = 1;
const EXT_FUTIL_MARGIN: i32 = ROOK_PTS;
const EXT_FUTIL_DEPTH: i8 = 2;
const RAZOR_MARGIN: i32 = QUEEN_PTS;
const RAZORING_DEPTH: i8 = 3;
/// Principal variation search - uses reduced alpha beta windows around a likely best move candidate
/// to refute other variations
fn pvs(
    mut depth: i8,
    mut alpha: i32,
    mut beta: i32,
    pv: &mut Vec<Move>,
    search_info: &mut SearchInfo,
    board: &Board,
) -> i32 {
    let ply = search_info.iter_max_depth - depth;
    let is_root = ply == 0;
    search_info.sel_depth = search_info.sel_depth.max(ply);
    // Don't do pvs unless you have a pv - otherwise you're wasting time
    let mut do_pvs = false;
    // Needed since the function can calculate extensions in cases where it finds itself in check
    if ply >= MAX_SEARCH_DEPTH {
        return eval(board);
    }

    if ply > 0 {
        if board.is_draw() {
            return STALEMATE;
        }

        // Determines if there is a faster path to checkmate than evaluating the current node, and
        // if there is, it returns early
        alpha = alpha.max(-CHECKMATE + ply as i32);
        beta = beta.min(CHECKMATE - ply as i32);
        if alpha >= beta {
            return alpha;
        }
    }

    let (table_value, table_move) = {
        let hash = board.zobrist_hash;
        let entry = search_info.transpos_table.get(&hash);
        if let Some(entry) = entry {
            entry.get(depth, ply, alpha, beta)
        } else {
            (None, Move::NULL)
        }
    };
    if let Some(eval) = table_value {
        if !is_root {
            return eval;
        }
    }

    let extend = board.side_in_check(board.to_move);

    if extend {
        depth += 1;
    }

    if depth <= 0 {
        return quiescence(ply, alpha, beta, pv, search_info, board);
    }

    search_info.search_stats.nodes_searched += 1;

    let mut moves = generate_psuedolegal_moves(board);
    let mut legal_moves = 0;
    score_move_list(ply, board, &mut moves, table_move, search_info);

    let mut score = -INFINITY;
    let mut entry_flag = EntryFlag::AlphaCutOff;
    let mut best_move = Move::NULL;
    let mut fprune = false;
    let mut pruned_moves = 0;
    let mut fmax = -INFINITY;

    let fscore = board.material_balance() + RAZOR_MARGIN;
    if !extend && depth == RAZORING_DEPTH && fscore <= alpha {
        fprune = true;
        fmax = fscore;
        score = fscore
    }

    let fscore = board.material_balance() + EXT_FUTIL_MARGIN;
    if !extend && depth == EXT_FUTIL_DEPTH && fscore <= alpha {
        fprune = true;
        fmax = fscore;
        score = fscore;
    }

    let fscore = board.material_balance() + FUTIL_MARGIN;
    if !extend && depth == FUTIL_DEPTH && fscore <= alpha {
        fprune = true;
        fmax = fscore;
        score = fscore;
        depth -= 1;
    }

    //  Null pruning from the psuedocode given by the chess programming wiki AEL pruning page
    //  if (!fprune && !check(move) && null_okay(current, move) &&
    //     try_null(alpha, beta, current, depth, move, tt_ref)) {
    //     int null_score;
    //     null_score = -alpha_beta_search(-beta, -beta + 1, null_move, current, depth - R_adpt(current, depth) - 1);
    //     if (null_score >= beta) {
    //         return null_score;
    //     }
    // }

    for i in 0..moves.len {
        let mut new_b = board.to_owned();
        sort_next_move(&mut moves, i);
        let m = moves.get_move(i);
        new_b.make_move(m);
        let _s = m.to_lan();
        let _c = moves.get_score(i);
        // Just generate psuedolegal moves to save computation time on legality for moves that will be
        // pruned
        if new_b.side_in_check(board.to_move) {
            continue;
        }
        legal_moves += 1;

        let capture = board.piece_on_square(m.dest_square());
        if fprune && fmax + value(capture) < alpha {
            pruned_moves += 1;
            continue;
        }

        let mut node_pvs = Vec::new();
        let mut eval;

        if do_pvs {
            eval = -pvs(
                depth - 1,
                -alpha - 1,
                -alpha,
                &mut node_pvs,
                search_info,
                &new_b,
            );
            if eval > alpha && alpha < beta {
                eval = -pvs(depth - 1, -beta, -alpha, &mut node_pvs, search_info, &new_b);
            }
        } else {
            eval = -pvs(depth - 1, -beta, -alpha, &mut node_pvs, search_info, &new_b);
        }

        if eval > score {
            score = eval;
            best_move = *m;
        }

        if eval >= beta {
            search_info.transpos_table.insert(
                board.zobrist_hash,
                TableEntry::new(depth, ply, EntryFlag::BetaCutOff, eval, best_move),
            );
            let capture = board.piece_on_square(m.dest_square());

            // Store a killer move if it is not a capture, but good enough to cause a beta cutoff
            // anyway
            if capture.is_none() {
                store_killer_move(ply, m, search_info);
            }
            return eval;
        }

        if eval > alpha {
            alpha = eval;
            entry_flag = EntryFlag::Exact;
            // A principal variation has been found, so we can do pvs on the remaining nodes of this level
            do_pvs = true;
            pv.clear();
            pv.push(*m);
            pv.append(&mut node_pvs);
        }
    }

    if legal_moves == 0 {
        // Checkmate
        if board.side_in_check(board.to_move) {
            // Distance from root is returned in order for other recursive calls to determine
            // shortest viable checkmate path
            return -CHECKMATE + ply as i32;
        }
        return STALEMATE;
    }

    search_info.transpos_table.insert(
        board.zobrist_hash,
        TableEntry::new(depth, ply, entry_flag, alpha, best_move),
    );

    if pruned_moves > 0 {
        // Handle selective pruning
    }

    alpha
}

// pub(super) fn score_move(board: &Board, m: &Move, search_info: &mut SearchInfo) -> u32 {
//     let mut score = 0;
//     let piece_moving = board
//         .piece_on_square(m.origin_square())
//         .expect("There should be a piece here");
//     let capture = board.piece_on_square(m.dest_square());
//     if let Some(capture) = capture {
//         score += 10 * capture.value() - piece_moving.value();
//     }
//     (score
//         + match m.promotion() {
//         Some(Promotion::Queen) => PieceName::Queen.value(),
//         Some(Promotion::Rook) => PieceName::Rook.value(),
//         Some(Promotion::Bishop) => PieceName::Bishop.value(),
//         Some(Promotion::Knight) => PieceName::Knight.value(),
//         None => 0,
//     }) as u32
// }

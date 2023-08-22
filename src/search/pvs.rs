use std::cmp::{max, min};
use std::time::{Duration, Instant};

use crate::board::board::Board;
use crate::engine::transposition::{EntryFlag, TableEntry};
use crate::moves::movegenerator::generate_psuedolegal_moves;
use crate::moves::movelist::MoveList;
use crate::moves::moves::Move;
use crate::moves::moves::Promotion;
use crate::types::pieces::PieceName;

use super::eval::eval;
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
        alpha = max(alpha, -CHECKMATE + ply as i32);
        beta = min(beta, CHECKMATE - ply as i32);
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
        // TODO: Try removing this if statement
        if !is_root {
            return eval;
        }
    }

    let is_check = board.side_in_check(board.to_move);

    if is_check {
        depth += 1;
    }

    if depth <= 0 {
        return quiescence(ply, alpha, beta, pv, search_info, board);
    }

    search_info.search_stats.nodes_searched += 1;

    // TODO: Test just generating all moves - I sort of hate how the code looks with psuedolegal
    // generation...
    let mut moves = generate_psuedolegal_moves(board);
    let mut legal_moves = 0;
    score_move_list(board, &mut moves, table_move);

    let mut best_eval = -INFINITY;
    let mut entry_flag = EntryFlag::AlphaCutOff;
    let mut best_move = Move::NULL;

    // This assumes the first move is the best, generates an evaluation, and then the future moves
    // are compared against this one

    for i in 0..moves.len {
        let mut new_b = board.to_owned();
        sort_next_move(&mut moves, i);
        let m = moves.get_move(i);
        new_b.make_move(m);
        // Just generate psuedolegal moves to save computation time on legality for moves that will be
        // pruned
        if new_b.side_in_check(board.to_move) {
            continue;
        }
        legal_moves += 1;

        let mut node_pvs = Vec::new();

        let mut eval = -pvs(
            depth - 1,
            -alpha - 1,
            -alpha,
            &mut node_pvs,
            search_info,
            &new_b,
        );
        if eval > alpha && eval < beta {
            eval = -pvs(depth - 1, -beta, -alpha, &mut node_pvs, search_info, &new_b);
            if eval > alpha {
                alpha = eval;
                entry_flag = EntryFlag::Exact;
                pv.clear();
                pv.push(*m);
                pv.append(&mut node_pvs);
            }
        }

        if eval > best_eval {
            if eval >= beta {
                search_info.transpos_table.insert(
                    board.zobrist_hash,
                    TableEntry::new(depth, ply, EntryFlag::BetaCutOff, eval, best_move),
                );
                return eval;
            }
            best_eval = eval;
            best_move = *m;
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

    alpha
}

pub(super) fn score_move(board: &Board, m: &Move) -> i32 {
    let mut score = 0;
    let piece_moving = board
        .piece_on_square(m.origin_square())
        .expect("There should be a piece here");
    let capture = board.piece_on_square(m.dest_square());
    if let Some(capture) = capture {
        score += 10 * capture.value() - piece_moving.value();
    }
    score
        + match m.promotion() {
            Some(Promotion::Queen) => PieceName::Queen.value(),
            Some(Promotion::Rook) => PieceName::Rook.value(),
            Some(Promotion::Bishop) => PieceName::Bishop.value(),
            Some(Promotion::Knight) => PieceName::Knight.value(),
            None => 0,
        }
}

const TT_BONUS: i32 = 500;
pub fn score_move_list(board: &Board, moves: &mut MoveList, table_move: Move) {
    for i in 0..moves.len {
        let (m, m_score) = moves.get_mut(i);
        let mut score = score_move(board, m);
        if table_move != Move::NULL && table_move == *m {
            score += TT_BONUS;
        }
        *m_score = score;
    }
}

pub fn sort_next_move(moves: &mut MoveList, idx: usize) {
    for i in (idx + 1)..moves.len {
        if moves.get_score(i) > moves.get_score(i) {
            moves.swap(idx, i);
        }
    }
}

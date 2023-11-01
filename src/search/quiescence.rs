use crate::board::board::Board;
use crate::engine::transposition::{EntryFlag, TableEntry};
use crate::eval::eval::evaluate;
use crate::moves::movegenerator::{generate_moves, MGT};
use crate::moves::movelist::MoveListEntry;
use crate::moves::moves::Move;
use crate::search::search::STALEMATE;

use super::search::{CHECKMATE, INFINITY};
use super::see::see;
use super::store_pv;
use super::{search::MAX_SEARCH_DEPTH, SearchInfo};

pub fn quiescence(
    ply: i32,
    mut alpha: i32,
    beta: i32,
    pvs: &mut Vec<Move>,
    info: &mut SearchInfo,
    board: &Board,
) -> i32 {
    if board.is_draw() {
        return STALEMATE;
    }

    info.sel_depth = info.sel_depth.max(ply);
    info.search_stats.nodes_searched += 1;

    if ply >= MAX_SEARCH_DEPTH {
        return evaluate(board);
    }

    let (_, table_move) = {
        // Returning an eval from this has been uncooperative, but we can definitely get a best move
        if let Some(entry) = info.transpos_table.read().unwrap().get(&board.zobrist_hash) {
            entry.get(0, ply, alpha, beta, board)
        } else {
            (None, Move::NULL)
        }
    };

    // Give the engine the chance to stop capturing here if it results in a better end result than continuing the chain of capturing
    let stand_pat = evaluate(board);
    if stand_pat >= beta {
        return stand_pat;
    }
    let original_alpha = alpha;
    alpha = alpha.max(stand_pat);

    let in_check = board.in_check(board.to_move);
    let mut moves = if in_check {
        generate_moves(board, MGT::All)
    } else {
        generate_moves(board, MGT::CapturesOnly)
    };
    moves.score_moves(board, table_move, &info.killer_moves[ply as usize], info);
    let mut best_score = if in_check { -INFINITY } else { evaluate(board) };
    let mut best_move = Move::NULL;
    let mut moves_searched = 0;

    for MoveListEntry { m, .. } in moves {
        let mut node_pvs = Vec::new();
        let mut new_b = board.to_owned();

        // We want to find at least one evasion so we know we aren't in checkmate, so don't prune
        // moves before then
        if (!in_check || moves_searched > 1) && !see(board, m, 1) {
            continue;
        }

        if !new_b.make_move(m) {
            continue;
        }
        info.current_line.push(m);
        moves_searched += 1;

        // TODO: Implement delta pruning

        let eval = -quiescence(ply + 1, -beta, -alpha, &mut node_pvs, info, &new_b);

        info.current_line.pop();

        if eval > best_score {
            best_score = eval;
            best_move = m;
            if eval > alpha {
                alpha = eval;
                store_pv(pvs, &mut node_pvs, m);
            }
            if alpha >= beta {
                break;
            }
        }
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
        .insert(board.zobrist_hash, TableEntry::new(0, ply, entry_flag, best_score, best_move));

    if in_check && moves_searched == 0 {
        return -CHECKMATE + ply;
    }

    best_score
}

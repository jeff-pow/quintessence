use strum::IntoEnumIterator;

use crate::{
    board::board::Board,
    moves::moves::{Move, Promotion},
    types::{
        bitboard::Bitboard,
        pieces::{Color, PieceName},
    },
};

fn gain(board: &Board, m: &Move) -> i32 {
    if m.is_castle() {
        return 0;
    }
    if m.is_en_passant() {
        return PieceName::Pawn.value();
    }
    let mut score = if let Some(capture) = board.piece_at(m.dest_square()) {
        capture.value()
    } else {
        0
    };
    if let Some(p) = m.promotion() {
        score += match p {
            Promotion::Queen => PieceName::Queen.value(),
            Promotion::Rook => PieceName::Rook.value(),
            Promotion::Bishop => PieceName::Bishop.value(),
            Promotion::Knight => PieceName::Knight.value(),
        };
        score -= PieceName::Pawn.value()
    }
    score
}

fn next_attacker(board: &Board, occupied: &mut Bitboard, attackers: Bitboard, side: Color) -> PieceName {
    for p in PieceName::iter().rev() {
        let mut bb = attackers & board.bitboard(side, p);
        if bb != Bitboard::EMPTY {
            *occupied ^= bb.pop_lsb().bitboard();
            return p;
        }
    }
    unreachable!()
}

pub fn see(board: &Board, m: &Move, threshold: i32) -> bool {
    let dest = m.dest_square();
    let src = m.origin_square();

    let mut val = gain(board, m) - threshold;
    if val < 0 {
        return false;
    }

    let piece_moving = board.piece_at(src).expect("There is a piece here");
    val -= piece_moving.value();
    if val >= 0 {
        return true;
    }

    let mut occupied = (board.occupancies() ^ src.bitboard()) | dest.bitboard();
    let mut attackers = board.attackers(dest, occupied) & occupied;

    let queens = board.bitboard(Color::White, PieceName::Queen) | board.bitboard(Color::Black, PieceName::Queen);
    let bishops =
        board.bitboard(Color::White, PieceName::Bishop) | board.bitboard(Color::Black, PieceName::Bishop) | queens;
    let rooks = board.bitboard(Color::White, PieceName::Rook) | board.bitboard(Color::Black, PieceName::Rook) | queens;

    let mut to_move = board.to_move.opp();

    loop {
        let my_attackers = attackers & board.color_occupancies(to_move);

        if my_attackers == Bitboard::EMPTY {
            break;
        }

        let next_piece = next_attacker(board, &mut occupied, my_attackers, to_move);

        if next_piece == PieceName::Pawn || next_piece == PieceName::Bishop || next_piece == PieceName::Queen {
            attackers |= board.mg.bishop_attacks(dest, occupied) & bishops;
        }
        if next_piece == PieceName::Rook || next_piece == PieceName::Queen {
            attackers |= board.mg.rook_attacks(dest, occupied) & rooks;
        }

        attackers &= occupied;

        to_move = to_move.opp();
        val = -val - 1 - next_piece.value();
        if val >= 0 {
            if next_piece == PieceName::King && (attackers & board.color_occupancies(to_move) != Bitboard::EMPTY) {
                to_move = to_move.opp();
            }
            break;
        }
    }

    to_move != board.to_move
}

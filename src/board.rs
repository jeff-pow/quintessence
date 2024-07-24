use core::fmt;

use super::fen::STARTING_FEN;
use crate::{
    attack_boards::{king_attacks, knight_attacks, pawn_attacks, pawn_set_attacks, BETWEEN_SQUARES, RANKS},
    chess_move::{
        Castle,
        Direction::{North, South},
        Move, MoveType, CASTLING_RIGHTS,
    },
    magics::{bishop_attacks, queen_attacks, rook_attacks},
    types::{
        bitboard::Bitboard,
        pieces::{Color, Piece, PieceName, NUM_PIECES},
        square::Square,
    },
    zobrist::ZOBRIST,
};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Board {
    bitboards: [Bitboard; NUM_PIECES],
    color_occupancies: [Bitboard; 2],
    mailbox: [Piece; 64],
    /// Side to move
    pub stm: Color,
    pub castling_rights: u32,
    pub en_passant_square: Option<Square>,
    pub num_moves: u16,
    pub half_moves: u16,
    pub zobrist_hash: u64,
    pub pawn_hash: u64,
    threats: Bitboard,
    checkers: Bitboard,
    pinned: Bitboard,
}

impl Default for Board {
    fn default() -> Self {
        Self::from_fen(STARTING_FEN)
    }
}

impl Board {
    pub fn piece_bbs(&self) -> [Bitboard; 6] {
        self.bitboards
    }

    pub fn color_bbs(&self) -> [Bitboard; 2] {
        self.color_occupancies
    }

    pub fn piece_color(&self, side: Color, piece: PieceName) -> Bitboard {
        self.piece(piece) & self.color(side)
    }

    pub fn piece(&self, piece: PieceName) -> Bitboard {
        self.bitboards[piece]
    }

    pub fn color(&self, color: Color) -> Bitboard {
        self.color_occupancies[color]
    }

    pub fn occupancies(&self) -> Bitboard {
        self.color(Color::White) | self.color(Color::Black)
    }

    pub fn piece_at(&self, sq: Square) -> Piece {
        self.mailbox[sq]
    }

    fn is_material_draw(&self) -> bool {
        // If we have any pawns, checkmate is still possible
        if self.piece(PieceName::Pawn) != Bitboard::EMPTY {
            return false;
        }
        let piece_count = self.occupancies().count_bits();
        // King vs King can't checkmate
        if piece_count == 2
               // If there's three pieces and a singular knight or bishop, stalemate is impossible
            || (piece_count == 3 && ((self.piece(PieceName::Knight).count_bits() == 1)
            || (self.piece(PieceName::Bishop).count_bits() == 1)))
        {
            return true;
        }
        if piece_count == 4 {
            // No combination of two knights and a king can checkmate
            if self.piece(PieceName::Knight).count_bits() == 2 {
                return true;
            }
            // If there is one bishop per side, checkmate is impossible
            if self.color(Color::White).count_bits() == 2 && self.piece(PieceName::Bishop).count_bits() == 2 {
                return true;
            }
        }

        false
    }

    pub fn hash_after(&self, m: Option<Move>) -> u64 {
        let mut hash = self.zobrist_hash ^ ZOBRIST.turn;

        // Return hash right away if the move was a null move
        let Some(m) = m else { return hash };

        hash ^= ZOBRIST.piece[m.piece_moving()][m.from()] ^ ZOBRIST.piece[m.piece_moving()][m.to()];

        if self.piece_at(m.to()) != Piece::None {
            hash ^= ZOBRIST.piece[self.piece_at(m.to())][m.to()];
        }

        hash
    }

    /// Returns the type of piece captured by a move, if any
    pub fn capture(&self, m: Move) -> Piece {
        if m.is_en_passant() {
            Piece::new(PieceName::Pawn, !self.stm)
        } else {
            self.piece_at(m.to())
        }
    }

    pub fn is_draw(&self) -> bool {
        self.half_moves >= 100 || self.is_material_draw()
    }

    pub fn has_non_pawns(&self, side: Color) -> bool {
        self.occupancies() ^ self.piece_color(side, PieceName::King) ^ self.piece_color(side, PieceName::Pawn)
            != Bitboard::EMPTY
    }

    pub const fn can_en_passant(&self) -> bool {
        self.en_passant_square.is_some()
    }

    pub const fn can_castle(&self, c: Castle) -> bool {
        match c {
            Castle::WhiteKing => self.castling_rights & Castle::WhiteKing as u32 != 0,
            Castle::WhiteQueen => self.castling_rights & Castle::WhiteQueen as u32 != 0,
            Castle::BlackKing => self.castling_rights & Castle::BlackKing as u32 != 0,
            Castle::BlackQueen => self.castling_rights & Castle::BlackQueen as u32 != 0,
            Castle::None => panic!(),
        }
    }

    pub fn place_piece(&mut self, piece: Piece, sq: Square) {
        self.mailbox[sq] = piece;
        self.bitboards[piece.name()] ^= sq.bitboard();
        self.color_occupancies[piece.color()] ^= sq.bitboard();
        self.zobrist_hash ^= ZOBRIST.piece[piece][sq];
        if piece.name() == PieceName::Pawn {
            self.pawn_hash ^= ZOBRIST.piece[piece][sq];
        }
    }

    fn remove_piece(&mut self, sq: Square) {
        let piece = self.mailbox[sq];
        if piece != Piece::None {
            self.mailbox[sq] = Piece::None;
            self.bitboards[piece.name()] ^= sq.bitboard();
            self.color_occupancies[piece.color()] ^= sq.bitboard();
            self.zobrist_hash ^= ZOBRIST.piece[piece][sq];
            if piece.name() == PieceName::Pawn {
                self.pawn_hash ^= ZOBRIST.piece[piece][sq];
            }
        }
    }

    pub fn king_square(&self, color: Color) -> Square {
        self.piece_color(color, PieceName::King).lsb()
    }

    pub fn attackers(&self, sq: Square, occupancy: Bitboard) -> Bitboard {
        self.attackers_for_side(Color::White, sq, occupancy) | self.attackers_for_side(Color::Black, sq, occupancy)
    }

    pub fn attackers_for_side(&self, attacker: Color, sq: Square, occupancy: Bitboard) -> Bitboard {
        let bishops = self.piece(PieceName::Queen) | self.piece(PieceName::Bishop);
        let rooks = self.piece(PieceName::Queen) | self.piece(PieceName::Rook);
        let pawn_attacks = pawn_attacks(sq, !attacker) & self.piece(PieceName::Pawn);
        let knight_attacks = knight_attacks(sq) & self.piece(PieceName::Knight);
        let bishop_attacks = bishop_attacks(sq, occupancy) & bishops;
        let rook_attacks = rook_attacks(sq, occupancy) & rooks;
        let king_attacks = king_attacks(sq) & self.piece(PieceName::King);
        (pawn_attacks | knight_attacks | bishop_attacks | rook_attacks | king_attacks) & self.color(attacker)
    }

    pub fn square_under_attack(&self, attacker: Color, sq: Square) -> bool {
        self.attackers_for_side(attacker, sq, self.occupancies()) != Bitboard::EMPTY
    }

    pub fn in_check(&self) -> bool {
        self.checkers() != Bitboard::EMPTY
    }

    pub const fn checkers(&self) -> Bitboard {
        self.checkers
    }

    pub const fn pinned(&self) -> Bitboard {
        self.pinned
    }

    pub(super) fn pinned_and_checkers(&mut self) {
        self.pinned = Bitboard::EMPTY;
        let attacker = !self.stm;
        let king_sq = self.king_square(self.stm);

        self.checkers = knight_attacks(king_sq) & self.piece_color(attacker, PieceName::Knight)
            | pawn_attacks(king_sq, self.stm) & self.piece_color(attacker, PieceName::Pawn);

        let sliders_attacks = self.diags(attacker) & bishop_attacks(king_sq, Bitboard::EMPTY)
            | self.orthos(attacker) & rook_attacks(king_sq, Bitboard::EMPTY);
        for sq in sliders_attacks {
            let blockers = BETWEEN_SQUARES[sq][king_sq] & self.occupancies();
            if blockers == Bitboard::EMPTY {
                // No pieces between attacker and king
                self.checkers |= sq.bitboard();
            } else if blockers.count_bits() == 1 {
                // One piece between attacker and king
                self.pinned |= blockers & self.color(self.stm);
            }
            // Multiple pieces between attacker and king, we don't really care
        }
    }

    pub(crate) fn diags(&self, side: Color) -> Bitboard {
        self.piece_color(side, PieceName::Bishop) | self.piece_color(side, PieceName::Queen)
    }

    pub(crate) fn orthos(&self, side: Color) -> Bitboard {
        self.piece_color(side, PieceName::Rook) | self.piece_color(side, PieceName::Queen)
    }

    pub(crate) const fn threats(&self) -> Bitboard {
        self.threats
    }

    pub(crate) fn calculate_threats(&mut self) {
        let attacker = !self.stm;
        let mut threats = Bitboard::EMPTY;
        let occ = self.occupancies() ^ self.king_square(self.stm).bitboard();

        threats |= pawn_set_attacks(self.piece_color(attacker, PieceName::Pawn), attacker);

        let rooks = (self.piece(PieceName::Rook) | self.piece(PieceName::Queen)) & self.color(attacker);
        rooks.into_iter().for_each(|sq| threats |= rook_attacks(sq, occ));

        let bishops = (self.piece(PieceName::Bishop) | self.piece(PieceName::Queen)) & self.color(attacker);
        bishops.into_iter().for_each(|sq| threats |= bishop_attacks(sq, occ));

        self.piece_color(attacker, PieceName::Knight).into_iter().for_each(|sq| threats |= knight_attacks(sq));

        threats |= king_attacks(self.king_square(attacker));

        self.threats = threats;
    }

    pub(crate) fn is_pseudo_legal(&self, m: Option<Move>) -> bool {
        let Some(m) = m else { return false };

        let from = m.from();
        let to = m.to();

        let moved_piece = self.piece_at(from);
        let captured_piece = self.piece_at(to);
        let is_capture = captured_piece != Piece::None;
        let is_pawn_double_push = m.flag() == MoveType::DoublePush;

        if moved_piece != m.piece_moving() {
            return false;
        }

        if moved_piece == Piece::None {
            return false;
        }

        if moved_piece.color() != self.stm {
            return false;
        }

        if is_capture && captured_piece.color() == self.stm {
            return false;
        }

        if m.is_castle() {
            if self.in_check() {
                return false;
            }
            if moved_piece.name() != PieceName::King {
                return false;
            }
            let castle = m.castle_type();
            if !self.can_castle(castle) {
                return false;
            }

            if self.occupancies() & castle.empty_squares() != Bitboard::EMPTY {
                return false;
            }
            if castle.check_squares() & self.threats() != Bitboard::EMPTY {
                return false;
            }
            if self.piece_color(self.stm, PieceName::Rook) & castle.rook_from().bitboard() == Bitboard::EMPTY {
                return false;
            }

            return true;
        }

        match moved_piece.name() {
            PieceName::Pawn => {
                if is_capture && is_pawn_double_push {
                    return false;
                }
                let bitboard = RANKS[7];
                let should_be_promoting =
                    to.bitboard() & bitboard != Bitboard::EMPTY || to.bitboard() & RANKS[0] != Bitboard::EMPTY;
                if should_be_promoting && m.promotion().is_none() {
                    return false;
                }
                let up = match self.stm {
                    Color::White => North,
                    Color::Black => South,
                };
                if m.is_en_passant() {
                    return Some(to) == self.en_passant_square;
                }
                if is_pawn_double_push {
                    let one_forward = from.shift(up);
                    return self.piece_at(one_forward) == Piece::None && to == one_forward.shift(up);
                }
                if !is_capture {
                    return to == from.shift(up) && captured_piece == Piece::None;
                }
                // Captures
                (pawn_attacks(from, self.stm) & to.bitboard()) != Bitboard::EMPTY
            }
            PieceName::Knight => to.bitboard() & knight_attacks(from) != Bitboard::EMPTY,
            PieceName::Bishop => to.bitboard() & bishop_attacks(from, self.occupancies()) != Bitboard::EMPTY,
            PieceName::Rook => to.bitboard() & rook_attacks(from, self.occupancies()) != Bitboard::EMPTY,
            PieceName::Queen => to.bitboard() & queen_attacks(from, self.occupancies()) != Bitboard::EMPTY,
            PieceName::King => to.bitboard() & king_attacks(from) != Bitboard::EMPTY,
            PieceName::None => panic!(),
        }
    }

    /// Function makes a move and modifies board state to reflect the move that just happened.
    /// Returns true if a move was legal, and false if it was illegal.
    #[must_use]
    pub fn make_move(&mut self, m: Move) -> bool {
        let piece_moving = m.piece_moving();
        assert_eq!(piece_moving, self.piece_at(m.from()));
        let capture = self.capture(m);
        self.remove_piece(m.to());

        if m.promotion().is_none() {
            self.place_piece(piece_moving, m.to());
        }

        self.remove_piece(m.from());

        // Move rooks if a castle move is applied
        if m.is_castle() {
            let rook = Piece::new(PieceName::Rook, self.stm);
            self.place_piece(rook, m.castle_type().rook_to());
            self.remove_piece(m.castle_type().rook_from());
        } else if let Some(p) = m.promotion() {
            self.place_piece(p, m.to());
        } else if m.is_en_passant() {
            match self.stm {
                Color::White => {
                    self.remove_piece(m.to().shift(South));
                }
                Color::Black => {
                    self.remove_piece(m.to().shift(North));
                }
            }
        }

        // If we are in check after all pieces have been moved, this move is illegal and we return
        // false to denote so
        if !self.king_square(self.stm).is_valid() {
            return false;
        }
        if self.square_under_attack(!self.stm, self.king_square(self.stm)) {
            return false;
        }

        // Xor out the old en passant square hash
        if let Some(sq) = self.en_passant_square {
            self.zobrist_hash ^= ZOBRIST.en_passant[sq];
        }
        // If the end index of a move is 16 squares from the start (and a pawn moved), an en passant is possible
        self.en_passant_square = None;
        if m.flag() == MoveType::DoublePush {
            match self.stm {
                Color::White => {
                    self.en_passant_square = Some(m.to().shift(South));
                }
                Color::Black => {
                    self.en_passant_square = Some(m.to().shift(North));
                }
            }
        }
        // Xor in the new en passant square hash
        if let Some(sq) = self.en_passant_square {
            self.zobrist_hash ^= ZOBRIST.en_passant[sq];
        }

        // If a piece isn't captured and a pawn isn't moved, increment the half move clock.
        // Otherwise set it to zero

        if capture == Piece::None && piece_moving.name() != PieceName::Pawn {
            self.half_moves += 1;
        } else {
            self.half_moves = 0;
        }

        self.zobrist_hash ^= ZOBRIST.castling[self.castling_rights as usize];
        self.castling_rights &= CASTLING_RIGHTS[m.from()] & CASTLING_RIGHTS[m.to()];
        self.zobrist_hash ^= ZOBRIST.castling[self.castling_rights as usize];

        self.stm = !self.stm;
        self.zobrist_hash ^= ZOBRIST.turn;

        self.num_moves += 1;

        self.calculate_threats();
        self.pinned_and_checkers();

        // This move is valid, so we return true to denote this fact
        true
    }

    pub fn make_null_move(&mut self) {
        self.stm = !self.stm;
        self.zobrist_hash ^= ZOBRIST.turn;
        self.num_moves += 1;
        self.half_moves += 1;
        if let Some(sq) = self.en_passant_square {
            self.zobrist_hash ^= ZOBRIST.en_passant[sq];
        }
        self.en_passant_square = None;
        self.calculate_threats();
        self.pinned_and_checkers();
    }

    pub fn mat_scale(&self) -> i32 {
        700 + ((PieceName::Knight.value() * self.piece(PieceName::Knight).count_bits())
            + (PieceName::Bishop.value() * self.piece(PieceName::Bishop).count_bits())
            + (PieceName::Rook.value() * self.piece(PieceName::Rook).count_bits())
            + (PieceName::Queen.value() * self.piece(PieceName::Queen).count_bits()))
            / 32
    }

    pub fn debug_bitboards(&self) {
        for color in Color::iter() {
            for piece in PieceName::iter() {
                dbg!("{:?} {:?}", color, piece);
                dbg!(self.piece_color(color, piece));
                dbg!("\n");
            }
        }
    }

    pub fn empty() -> Self {
        Self {
            bitboards: [Bitboard::EMPTY; 6],
            color_occupancies: [Bitboard::EMPTY; 2],
            mailbox: [Piece::None; 64],
            castling_rights: 0,
            stm: Color::White,
            en_passant_square: None,
            num_moves: 0,
            pawn_hash: 0,
            half_moves: 0,
            zobrist_hash: 0,
            threats: Bitboard::EMPTY,
            checkers: Bitboard::EMPTY,
            pinned: Bitboard::EMPTY,
        }
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut str = String::new();

        for row in (0..8).rev() {
            str.push_str(&(row + 1).to_string());
            str.push_str(" | ");

            for col in 0..8 {
                let idx = row * 8 + col;

                // Append piece characters for white pieces
                let piece = self.piece_at(Square(idx));
                let char = piece.char();

                str += &char;

                str.push_str(" | ");
            }

            str.push('\n');
        }

        str.push_str("    a   b   c   d   e   f   g   h\n");

        str.push('\n');
        str.push_str(&self.to_fen());
        str.push('\n');

        write!(f, "{str}")
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut str = String::new();
        str += match self.stm {
            Color::White => "White to move\n",
            Color::Black => "Black to move\n",
        };
        str += &self.to_string();
        str += "threats:\n";
        str += &format!("{:?}\n", self.threats());
        str += "checkers:\n";
        str += &format!("{:?}\n", self.checkers);
        str += "pinned:\n";
        str += &format!("{:?}\n", self.pinned);
        str += "\n";
        str += "Castles available: ";
        if self.can_castle(Castle::WhiteKing) {
            str += "K";
        };
        if self.can_castle(Castle::WhiteQueen) {
            str += "Q";
        };
        if self.can_castle(Castle::BlackKing) {
            str += "k";
        };
        if self.can_castle(Castle::BlackQueen) {
            str += "q";
        };
        str += "\n";
        str += "En Passant Square: ";
        if let Some(s) = &self.en_passant_square {
            str += &s.to_string();
        } else {
            str += "None";
        }
        str += "\n";
        str += "Num moves made: ";
        str += &self.num_moves.to_string();
        str += "\n";

        write!(f, "{str}")
    }
}

#[cfg(test)]
mod board_tests {
    use super::*;
    #[test]
    fn test_place_piece() {
        let mut board = Board::empty();
        board.place_piece(Piece::WhiteRook, Square(0));
        assert!(board.piece_color(Color::White, PieceName::Rook).occupied(Square(0)));
    }

    #[test]
    fn test_remove_piece() {
        let board = Board::from_fen(STARTING_FEN);

        let mut c = board;
        c.remove_piece(Square(0));
        assert!(c.piece_color(Color::White, PieceName::Rook).empty(Square(0)));
        assert!(c.occupancies().empty(Square(0)));
        assert_ne!(c, board);

        let mut c = board;
        c.remove_piece(Square(27));
        assert_eq!(board, c);
    }
}

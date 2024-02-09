use crate::{
    board::board::Board,
    moves::{moves::Direction, moves::Direction::*},
    types::{
        bitboard::Bitboard,
        pieces::{Color, Piece, PieceName},
        square::Square,
    },
};

use super::{
    attack_boards::{king_attacks, knight_attacks, RANKS},
    magics::{bishop_attacks, queen_attacks, rook_attacks},
    movelist::MoveList,
    moves::{Castle, Move, MoveType},
};

#[allow(clippy::upper_case_acronyms)]
pub type MGT = MoveGenerationType;
#[derive(Copy, Clone, PartialEq)]
pub enum MoveGenerationType {
    CapturesOnly,
    QuietsOnly,
    All,
}

impl Board {
    /// Generates all pseudolegal moves
    pub fn generate_moves(&self, gen_type: MGT, moves: &mut MoveList) {
        let destinations = match gen_type {
            MoveGenerationType::CapturesOnly => self.color(!self.to_move),
            MoveGenerationType::QuietsOnly => !self.occupancies(),
            MoveGenerationType::All => !self.color(self.to_move),
        };

        let knights = self.bitboard(self.to_move, PieceName::Knight);
        let bishops = self.bitboard(self.to_move, PieceName::Bishop);
        let rooks = self.bitboard(self.to_move, PieceName::Rook);
        let queens = self.bitboard(self.to_move, PieceName::Queen);
        let kings = self.bitboard(self.to_move, PieceName::King);

        self.jumper_moves(knights, destinations, moves, knight_attacks);
        self.jumper_moves(kings, destinations & !self.threats(), moves, king_attacks);
        self.slider_moves(queens, destinations, self.occupancies(), moves, queen_attacks);
        self.slider_moves(rooks, destinations, self.occupancies(), moves, rook_attacks);
        self.slider_moves(bishops, destinations, self.occupancies(), moves, bishop_attacks);
        self.generate_pawn_moves(gen_type, moves);
        if gen_type == MGT::QuietsOnly || gen_type == MGT::All {
            self.generate_castling_moves(moves);
        }
    }

    fn generate_castling_moves(&self, moves: &mut MoveList) {
        if self.to_move == Color::White {
            if self.can_castle(Castle::WhiteKing)
                && self.threats() & Castle::WhiteKing.check_squares() == Bitboard::EMPTY
                && self.occupancies() & Castle::WhiteKing.empty_squares() == Bitboard::EMPTY
            {
                moves.push(Move::new(Square(4), Square(6), MoveType::CastleMove, Piece::WhiteKing));
            }
            if self.can_castle(Castle::WhiteQueen)
                && self.threats() & Castle::WhiteQueen.check_squares() == Bitboard::EMPTY
                && self.occupancies() & Castle::WhiteQueen.empty_squares() == Bitboard::EMPTY
            {
                moves.push(Move::new(Square(4), Square(2), MoveType::CastleMove, Piece::WhiteKing));
            }
        } else {
            if self.can_castle(Castle::BlackKing)
                && self.threats() & Castle::BlackKing.check_squares() == Bitboard::EMPTY
                && self.occupancies() & Castle::BlackKing.empty_squares() == Bitboard::EMPTY
            {
                moves.push(Move::new(
                    Square(60),
                    Square(62),
                    MoveType::CastleMove,
                    Piece::BlackKing,
                ));
            }
            if self.can_castle(Castle::BlackQueen)
                && self.threats() & Castle::BlackQueen.check_squares() == Bitboard::EMPTY
                && self.occupancies() & Castle::BlackQueen.empty_squares() == Bitboard::EMPTY
            {
                moves.push(Move::new(
                    Square(60),
                    Square(58),
                    MoveType::CastleMove,
                    Piece::BlackKing,
                ));
            }
        }
    }

    fn generate_pawn_moves(&self, gen_type: MGT, moves: &mut MoveList) {
        let piece = Piece::new(PieceName::Pawn, self.to_move);
        let pawns = self.bitboard(self.to_move, PieceName::Pawn);
        let vacancies = !self.occupancies();
        let enemies = self.color(!self.to_move);

        let non_promotions =
            pawns & if self.to_move == Color::White { !RANKS[6] } else { !RANKS[1] };
        let promotions = pawns & if self.to_move == Color::White { RANKS[6] } else { RANKS[1] };

        let up = if self.to_move == Color::White { North } else { South };
        let right = if self.to_move == Color::White { NorthEast } else { SouthWest };
        let left = if self.to_move == Color::White { NorthWest } else { SouthEast };

        let rank3 = if self.to_move == Color::White { RANKS[2] } else { RANKS[5] };

        if matches!(gen_type, MGT::All | MGT::QuietsOnly) {
            // Single and double pawn pushes w/o captures
            let push_one = vacancies & non_promotions.shift(up);
            let push_two = vacancies & (push_one & rank3).shift(up);
            for dest in push_one {
                let src = dest.shift(up.opp());
                moves.push(Move::new(src, dest, MoveType::Normal, piece));
            }
            for dest in push_two {
                let src = dest.shift(up.opp()).shift(up.opp());
                moves.push(Move::new(src, dest, MoveType::DoublePush, piece));
            }
        }

        // Promotions - captures and straight pushes
        // Promotions are generated with captures because they are so good
        if matches!(gen_type, MGT::All | MGT::CapturesOnly) && promotions != Bitboard::EMPTY {
            let no_capture_promotions = promotions.shift(up) & vacancies;
            let left_capture_promotions = promotions.shift(left) & enemies;
            let right_capture_promotions = promotions.shift(right) & enemies;
            for dest in no_capture_promotions {
                gen_promotions(piece, dest.shift(up.opp()), dest, moves);
            }
            for dest in left_capture_promotions {
                gen_promotions(piece, dest.shift(left.opp()), dest, moves);
            }
            for dest in right_capture_promotions {
                gen_promotions(piece, dest.shift(right.opp()), dest, moves);
            }
        }

        if matches!(gen_type, MGT::All | MGT::CapturesOnly) {
            // Captures that do not lead to promotions
            if non_promotions != Bitboard::EMPTY {
                let left_captures = non_promotions.shift(left) & enemies;
                let right_captures = non_promotions.shift(right) & enemies;
                for dest in left_captures {
                    let src = dest.shift(left.opp());
                    moves.push(Move::new(src, dest, MoveType::Normal, piece));
                }
                for dest in right_captures {
                    let src = dest.shift(right.opp());
                    moves.push(Move::new(src, dest, MoveType::Normal, piece));
                }
            }

            // En Passant
            if self.can_en_passant() {
                if let Some(x) = self.get_en_passant(left.opp(), piece) {
                    moves.push(x)
                }
                if let Some(x) = self.get_en_passant(right.opp(), piece) {
                    moves.push(x)
                }
            }
        }
    }

    fn get_en_passant(&self, dir: Direction, piece: Piece) -> Option<Move> {
        let sq = self.en_passant_square?.checked_shift(dir)?;
        let pawn = sq.bitboard() & self.bitboard(self.to_move, PieceName::Pawn);
        if pawn != Bitboard::EMPTY {
            let dest = self.en_passant_square?;
            let src = dest.checked_shift(dir)?;
            return Some(Move::new(src, dest, MoveType::EnPassant, piece));
        }
        None
    }

    fn generate_bitboard_moves(&self, piece_name: PieceName, gen_type: MGT, moves: &mut MoveList) {
        // Don't calculate any moves if no pieces of that type exist for the given color
        let occ_bitself = self.bitboard(self.to_move, piece_name);
        let piece_moving = Piece::new(piece_name, self.to_move);
        for sq in occ_bitself {
            let occupancies = self.occupancies();
            let attack_bitself = match piece_name {
                PieceName::King => king_attacks(sq) & !self.threats(),
                PieceName::Queen => queen_attacks(sq, occupancies),
                PieceName::Rook => rook_attacks(sq, occupancies),
                PieceName::Bishop => bishop_attacks(sq, occupancies),
                PieceName::Knight => knight_attacks(sq),
                _ => panic!(),
            };
            let attacks = match gen_type {
                MoveGenerationType::CapturesOnly => attack_bitself & self.color(!self.to_move),
                MoveGenerationType::QuietsOnly => attack_bitself & !self.occupancies(),
                MoveGenerationType::All => attack_bitself & (!self.color(self.to_move)),
            };
            for dest in attacks {
                moves.push(Move::new(sq, dest, MoveType::Normal, piece_moving));
            }
        }
    }
    fn slider_moves(
        &self,
        pieces: Bitboard,
        destinations: Bitboard,
        occupancies: Bitboard,
        moves: &mut MoveList,
        attack_fn: impl Fn(Square, Bitboard) -> Bitboard,
    ) {
        for src in pieces {
            for dest in attack_fn(src, occupancies) & destinations {
                moves.push(Move::new(src, dest, MoveType::Normal, self.piece_at(src)));
            }
        }
    }

    fn jumper_moves(
        &self,
        pieces: Bitboard,
        destinations: Bitboard,
        moves: &mut MoveList,
        attack_fn: impl Fn(Square) -> Bitboard,
    ) {
        for src in pieces {
            for dest in attack_fn(src) & destinations {
                moves.push(Move::new(src, dest, MoveType::Normal, self.piece_at(src)));
            }
        }
    }
}

fn gen_promotions(piece: Piece, src: Square, dest: Square, moves: &mut MoveList) {
    const PROMOS: [MoveType; 4] = [
        MoveType::QueenPromotion,
        MoveType::RookPromotion,
        MoveType::BishopPromotion,
        MoveType::KnightPromotion,
    ];
    for promo in PROMOS {
        moves.push(Move::new(src, dest, promo, piece));
    }
}

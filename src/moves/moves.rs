use core::fmt;
use std::fmt::Display;

use crate::{
    board::board::Board,
    moves::moves::Direction::*,
    types::{pieces::PieceName, square::Square},
};

use strum_macros::EnumIter;

#[derive(Clone, Copy)]
pub(crate) enum MoveType {
    Normal,
    Promotion,
    EnPassant,
    Castle,
}

fn get_move_type(promotion: bool, en_passant: bool, castle: bool) -> MoveType {
    if promotion {
        MoveType::Promotion
    } else if en_passant {
        return MoveType::EnPassant;
    } else if castle {
        return MoveType::Castle;
    } else {
        MoveType::Normal
    }
}

/// Cardinal directions from the point of view of white side
#[derive(EnumIter, Copy, Clone, Debug, PartialEq, Eq)]
pub enum Direction {
    North = 8,
    NorthWest = 7,
    West = -1,
    SouthWest = -9,
    South = -8,
    SouthEast = -7,
    East = 1,
    NorthEast = 9,
}

impl Direction {
    /// Returns the opposite direction of the given direction
    pub fn opp(self) -> Self {
        match self {
            North => South,
            NorthWest => SouthEast,
            West => East,
            SouthWest => NorthEast,
            South => North,
            SouthEast => NorthWest,
            East => West,
            NorthEast => SouthWest,
        }
    }

    pub fn to_xy(self) -> (i8, i8) {
        match self {
            North => (0, 1),
            NorthWest => (-1, 1),
            West => (-1, 0),
            SouthWest => (-1, -1),
            South => (0, -1),
            SouthEast => (1, -1),
            East => (1, 0),
            NorthEast => (1, 1),
        }
    }
}

/// A move needs 16 bits to be stored
///
/// bit  0- 5: origin square (from 0 to 63)
/// bit  6-11: destination square (from 0 to 63)
/// bit 12-13: promotion piece
/// bit 14-15: special move flag: normal move(0), promotion (1), en passant (2), castling (3)
/// bit 16-19: piece moving - useful in continuation history
/// NOTE: en passant bit is set only when a pawn can be captured
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Move(u32);

impl Move {
    pub const NULL: Move = Move(0);
    fn full_move(
        origin: Square,
        destination: Square,
        promotion: Option<Promotion>,
        move_type: MoveType,
        piece_moving: PieceName,
    ) -> Self {
        let promotion = match promotion {
            Some(Promotion::Queen) => 3,
            Some(Promotion::Rook) => 2,
            Some(Promotion::Bishop) => 1,
            Some(Promotion::Knight) | None => 0,
        };
        let move_type = match move_type {
            MoveType::Normal => 0,
            MoveType::Promotion => 1,
            MoveType::EnPassant => 2,
            MoveType::Castle => 3,
        };
        let piece = piece_moving.idx() as u32;
        let m = origin.0 as u32 | ((destination.0 as u32) << 6) | (promotion << 12) | (move_type << 14) | (piece << 16);
        Move(m)
    }

    pub fn new(origin: Square, destination: Square, piece_moving: PieceName) -> Self {
        Self::full_move(origin, destination, None, MoveType::Normal, piece_moving)
    }

    pub fn new_castling(origin: Square, destination: Square) -> Self {
        Self::full_move(origin, destination, None, MoveType::Castle, PieceName::King)
    }

    pub fn new_promotion(origin: Square, destination: Square, promotion: Promotion) -> Self {
        Self::full_move(origin, destination, Some(promotion), MoveType::Promotion, PieceName::Pawn)
    }

    pub fn new_en_passant(origin: Square, destination: Square) -> Self {
        Self::full_move(origin, destination, None, MoveType::EnPassant, PieceName::Pawn)
    }

    /// Creates a move from a raw u32. I hope you know what you're doing...
    pub fn raw(m: u32) -> Self {
        Self(m)
    }

    #[inline(always)]
    pub fn is_capture(self, board: &Board) -> bool {
        board.occupancies().square_occupied(self.dest_square())
    }

    #[inline(always)]
    pub fn is_castle(self) -> bool {
        let castle_flag = (self.0 >> 14) & 0b11;
        castle_flag == MoveType::Castle as u32
    }

    #[inline(always)]
    pub fn piece_moving(self) -> PieceName {
        let piece_flag = (self.0 >> 16) & 0b111;
        PieceName::from(piece_flag as usize)
    }

    #[inline(always)]
    pub fn is_en_passant(self) -> bool {
        let en_passant_flag = (self.0 >> 14) & 0b11;
        en_passant_flag == MoveType::EnPassant as u32
    }

    #[inline(always)]
    pub fn is_normal(self, board: &Board) -> bool {
        self.promotion().is_none()
            && !self.is_castle()
            && !self.is_en_passant()
            && !self.is_castle()
            && !self.is_capture(board)
    }

    #[inline(always)]
    pub fn promotion(self) -> Option<Promotion> {
        let promotion_flag = (self.0 >> 14) & 0b11;
        if promotion_flag != MoveType::Promotion as u32 {
            return None;
        }
        match (self.0 >> 12) & 0b11 {
            0 => Some(Promotion::Knight),
            1 => Some(Promotion::Bishop),
            2 => Some(Promotion::Rook),
            3 => Some(Promotion::Queen),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn origin_square(self) -> Square {
        Square((self.0 & 0b111111) as u8)
    }

    #[inline(always)]
    pub fn dest_square(self) -> Square {
        Square(((self.0 >> 6) & 0b111111) as u8)
    }

    #[inline(always)]
    pub fn as_u16(self) -> u16 {
        self.0 as u16
    }

    /// To Short Algebraic Notation
    pub fn to_san(self) -> String {
        let mut str = String::new();
        let arr = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let origin_number = self.origin_square().rank() + 1;
        let origin_letter = self.origin_square().file();
        let end_number = self.dest_square().rank() + 1;
        let end_letter = self.dest_square().file();
        str += arr[origin_letter as usize];
        str += &origin_number.to_string();
        str += arr[end_letter as usize];
        str += &end_number.to_string();
        match self.promotion() {
            Some(Promotion::Queen) => str += "q",
            Some(Promotion::Rook) => str += "r",
            Some(Promotion::Bishop) => str += "b",
            Some(Promotion::Knight) => str += "n",
            None => (),
        }
        str
    }

    #[inline(always)]
    pub fn castle_type(self) -> Castle {
        debug_assert!(self.is_castle());
        if self.dest_square().dist(self.origin_square()) != 2 {
            Castle::None
        } else if self.dest_square() == Square(2) {
            Castle::WhiteQueen
        } else if self.dest_square() == Square(6) {
            Castle::WhiteKing
        } else if self.dest_square() == Square(58) {
            Castle::BlackQueen
        } else if self.dest_square() == Square(62) {
            Castle::BlackKing
        } else {
            unreachable!()
        }
    }
}

/// Method converts a san move provided by UCI framework into a Move struct
pub fn from_san(str: &str, board: &Board) -> Move {
    let vec: Vec<char> = str.chars().collect();

    // Using base 20 allows program to convert letters directly to numbers instead of matching
    // against letters or some other workaround
    let start_column = vec[0].to_digit(20).unwrap() - 10;
    let start_row = (vec[1].to_digit(10).unwrap() - 1) * 8;
    let origin_sq = Square((start_row + start_column) as u8);

    let end_column = vec[2].to_digit(20).unwrap() - 10;
    let end_row = (vec[3].to_digit(10).unwrap() - 1) * 8;
    let dest_sq = Square((end_row + end_column) as u8);

    let promotion = if vec.len() > 4 {
        match vec[4] {
            'q' => Some(Promotion::Queen),
            'r' => Some(Promotion::Rook),
            'b' => Some(Promotion::Bishop),
            'n' => Some(Promotion::Knight),
            _ => panic!(),
        }
    } else {
        None
    };
    let piece_moving = board.piece_at(origin_sq).expect("There should be a piece here...");
    let captured = board.piece_at(dest_sq);
    let castle = match piece_moving {
        PieceName::King => {
            if origin_sq.dist(dest_sq) != 2 {
                Castle::None
            } else if dest_sq == Square(2) {
                Castle::WhiteQueen
            } else if dest_sq == Square(6) {
                Castle::WhiteKing
            } else if dest_sq == Square(58) {
                Castle::BlackQueen
            } else if dest_sq == Square(62) {
                Castle::BlackKing
            } else {
                unreachable!()
            }
        }
        _ => Castle::None,
    };
    let castle = castle != Castle::None;
    let en_passant = { piece_moving == PieceName::Pawn && captured.is_none() && start_column != end_column };
    let move_type = get_move_type(promotion.is_some(), en_passant, castle);
    Move::full_move(origin_sq, dest_sq, promotion, move_type, piece_moving)
}

#[derive(Clone, Copy, Debug, EnumIter, PartialEq)]
pub enum Promotion {
    Queen,
    Rook,
    Bishop,
    Knight,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Castle {
    WhiteKing = 1,
    WhiteQueen = 2,
    BlackKing = 4,
    BlackQueen = 8,
    None,
}

#[rustfmt::skip]
pub const CASTLING_RIGHTS: [u8; 64] = [
    13, 15, 15, 15, 12, 15, 15, 14,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    7, 15, 15, 15, 3, 15, 15, 11,
];

impl Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut str = String::new();
        str += "Start: ";
        str += &self.origin_square().to_string();
        str += " End: ";
        str += &self.dest_square().to_string();
        str += " Castle: ";
        str += &self.is_castle().to_string();
        str += " Promotion: ";
        match self.promotion() {
            Some(Promotion::Queen) => str += "Queen ",
            Some(Promotion::Rook) => str += "Rook ",
            Some(Promotion::Bishop) => str += "Bishop ",
            Some(Promotion::Knight) => str += "Knight ",
            None => str += "None ",
        }
        str += " En Passant: ";
        str += &self.is_en_passant().to_string();
        str += "  ";
        str += &self.to_san();
        write!(f, "{}", str)
    }
}

#[cfg(test)]
mod move_test {
    use super::*;

    #[test]
    fn test_move_creation() {
        let normal_move = Move::full_move(Square(10), Square(20), None, MoveType::Normal, PieceName::Pawn);
        assert_eq!(normal_move.origin_square(), Square(10));
        assert_eq!(normal_move.dest_square(), Square(20));
        assert!(!normal_move.is_castle());
        assert!(!normal_move.is_en_passant());
        assert_eq!(normal_move.promotion(), None);

        let promotion_move =
            Move::full_move(Square(15), Square(25), Some(Promotion::Queen), MoveType::Promotion, PieceName::Pawn);
        assert_eq!(promotion_move.origin_square(), Square(15));
        assert_eq!(promotion_move.dest_square(), Square(25));
        assert!(!promotion_move.is_castle());
        assert!(!promotion_move.is_en_passant());
        assert_eq!(promotion_move.promotion(), Some(Promotion::Queen));

        let castle_move = Move::full_move(Square(4), Square(2), None, MoveType::Castle, PieceName::King);
        assert_eq!(castle_move.origin_square(), Square(4));
        assert_eq!(castle_move.dest_square(), Square(2));
        assert!(castle_move.is_castle());
        assert!(!castle_move.is_en_passant());
        assert_eq!(castle_move.promotion(), None);

        let en_passant_move = Move::full_move(Square(7), Square(5), None, MoveType::EnPassant, PieceName::King);
        assert_eq!(en_passant_move.origin_square(), Square(7));
        assert_eq!(en_passant_move.dest_square(), Square(5));
        assert!(!en_passant_move.is_castle());
        assert!(en_passant_move.is_en_passant());
        assert_eq!(en_passant_move.promotion(), None);
    }

    #[test]
    fn test_promotion_conversion() {
        let knight_promotion =
            Move::full_move(Square(0), Square(7), Some(Promotion::Knight), MoveType::Promotion, PieceName::Pawn);
        assert_eq!(knight_promotion.promotion(), Some(Promotion::Knight));

        let bishop_promotion =
            Move::full_move(Square(15), Square(23), Some(Promotion::Bishop), MoveType::Promotion, PieceName::Pawn);
        assert_eq!(bishop_promotion.promotion(), Some(Promotion::Bishop));

        let rook_promotion =
            Move::full_move(Square(28), Square(31), Some(Promotion::Rook), MoveType::Promotion, PieceName::Pawn);
        assert_eq!(rook_promotion.promotion(), Some(Promotion::Rook));

        let queen_promotion =
            Move::full_move(Square(62), Square(61), Some(Promotion::Queen), MoveType::Promotion, PieceName::Pawn);
        assert_eq!(queen_promotion.promotion(), Some(Promotion::Queen));
    }
}

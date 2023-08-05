use core::fmt;
use std::fmt::Display;

use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::attack_boards::{king_attacks, knight_attacks, RANK2, RANK3, RANK6, RANK7};
use crate::bit_hacks::*;
use crate::magics::{get_bishop_moves, get_rook_moves};
// use crate::pieces::PieceName::{Bishop, King, Knight, Pawn, Queen, Rook};
use crate::pieces::opposite_color;
use crate::pieces::PieceName::Pawn;
use crate::{board::Board, pieces::Color, pieces::PieceName};

enum MoveType {
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
#[repr(i8)]
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

/// A move needs 16 bits to be stored
///
/// bit  0- 5: origin square (from 0 to 63)
/// bit  6-11: destination square (from 0 to 63)
/// bit 12-13: promotion piece
/// bit 14-15: special move flag: normal move(0), promotion (1), en passant (2), castling (3)
/// NOTE: en passant bit is set only when a pawn can be captured
#[derive(Clone, Copy, Debug)]
pub struct Move(u16);

impl Move {
    /// A move needs 16 bits to be stored
    ///
    /// bit  0- 5: origin square (from 0 to 63)
    /// bit  6-11: destination square (from 0 to 63)
    /// bit 12-13: promotion piece
    /// bit 14-15: special move flag: normal move (0), promotion (1), en passant (2), castling (3)
    /// NOTE: en passant bit is set only when a pawn can be captured
    fn new(origin: u8, destination: u8, promotion: Option<Promotion>, move_type: MoveType) -> Self {
        debug_assert!(origin < 64);
        debug_assert!(destination < 64);
        let promotion = match promotion {
            Some(Promotion::Queen) => 3,
            Some(Promotion::Rook) => 2,
            Some(Promotion::Bishop) => 1,
            Some(Promotion::Knight) => 0,
            None => 0,
        };
        let move_type = match move_type {
            MoveType::Normal => 0,
            MoveType::Promotion => 1,
            MoveType::EnPassant => 2,
            MoveType::Castle => 3,
        };
        let m = origin as u16 | ((destination as u16) << 6) | (promotion << 12) | (move_type << 14);
        Move(m)
    }

    #[inline]
    pub fn is_castle(&self) -> bool {
        let castle_flag = (self.0 >> 14) & 0b11;
        castle_flag == 3
    }

    #[inline]
    pub fn is_en_passant(&self) -> bool {
        let en_passant_flag = (self.0 >> 14) & 0b11;
        en_passant_flag == 2
    }

    #[inline]
    pub fn promotion(&self) -> Option<Promotion> {
        let promotion_flag = (self.0 >> 14) & 0b11;
        if promotion_flag != 1 {
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

    #[inline]
    pub fn origin_square(&self) -> u8 {
        (self.0 & 0b111111) as u8
    }

    #[inline]
    pub fn dest_square(&self) -> u8 {
        ((self.0 >> 6) & 0b111111) as u8
    }

    /// Determines if a move is "quiet" for quiescence search
    #[allow(dead_code)]
    #[inline]
    pub fn is_quiet(&self, board: &Board) -> bool {
        board.piece_on_square(self.dest_square().into()).is_none()
    }

    /// To Long Algebraic Notation
    pub fn to_lan(self) -> String {
        let mut str = String::new();
        let arr = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let y_origin = self.origin_square() / 8 + 1;
        let x_origin = self.origin_square() % 8;
        let y_end = self.dest_square() / 8 + 1;
        let x_end = self.dest_square() % 8;
        str += arr[x_origin as usize];
        str += &y_origin.to_string();
        str += arr[x_end as usize];
        str += &y_end.to_string();
        match self.promotion() {
            Some(Promotion::Queen) => str += "q",
            Some(Promotion::Rook) => str += "r",
            Some(Promotion::Bishop) => str += "b",
            Some(Promotion::Knight) => str += "n",
            None => (),
        }
        str
    }

    /// Constructor for new moves - Mostly a placeholder for initializing variables that will
    /// certainly be changed at some other point during the runtime of the function
    pub fn invalid() -> Self {
        Move::new(0, 0, None, MoveType::Normal)
    }
}

/// Method converts a lan move provided by UCI framework into a Move struct
pub fn from_lan(str: &str, board: &Board) -> Move {
    let vec: Vec<char> = str.chars().collect();

    // Using base 20 allows program to convert letters directly to numbers instead of matching
    // against letters or some other workaround
    let start_column = vec[0].to_digit(20).unwrap() - 10;
    let start_row = (vec[1].to_digit(10).unwrap() - 1) * 8;
    let starting_idx = (start_row + start_column) as u8;

    let end_column = vec[2].to_digit(20).unwrap() - 10;
    let end_row = (vec[3].to_digit(10).unwrap() - 1) * 8;
    let end_idx = (end_row + end_column) as u8;

    let mut promotion = None;
    if vec.len() > 4 {
        promotion = match vec[4] {
            'q' => Some(Promotion::Queen),
            'r' => Some(Promotion::Rook),
            'b' => Some(Promotion::Bishop),
            'n' => Some(Promotion::Knight),
            _ => panic!(),
        };
    }
    let piece_moving = board
        .piece_on_square(starting_idx as usize)
        .expect("There should be a piece here...");
    let castle = match piece_moving {
        PieceName::King => {
            if u8::abs_diff(starting_idx, end_idx) != 2 {
                Castle::None
            } else if end_idx == 2 {
                Castle::WhiteQueenCastle
            } else if end_idx == 6 {
                Castle::WhiteKingCastle
            } else if end_idx == 58 {
                Castle::BlackQueenCastle
            } else if end_idx == 62 {
                Castle::BlackKingCastle
            } else {
                unreachable!()
            }
        }
        _ => Castle::None,
    };
    let castle = castle != Castle::None;
    // TODO: Implement reading en passant...
    // BUG: Implement reading en passant...
    let en_passant = false;
    let move_type = get_move_type(promotion.is_some(), en_passant, castle);
    Move::new(starting_idx, end_idx, promotion, move_type)
}

#[derive(Clone, Copy, Debug, EnumIter, PartialEq)]
pub enum Promotion {
    Queen,
    Rook,
    Bishop,
    Knight,
}

#[derive(Clone, Copy, Debug, EnumIter, PartialEq)]
pub enum EnPassant {
    NW,
    NE,
    SW,
    SE,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Castle {
    None,
    WhiteKingCastle,
    WhiteQueenCastle,
    BlackKingCastle,
    BlackQueenCastle,
}

#[inline]
pub fn coordinates(idx: usize) -> (usize, usize) {
    (idx % 8, idx / 8)
}

pub fn generate_psuedolegal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::new();
    moves.append(&mut generate_bitboard_moves(board, PieceName::Knight));
    moves.append(&mut generate_bitboard_moves(board, PieceName::King));
    moves.append(&mut gen_pawn_moves(board));
    moves
}

fn gen_pawn_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::new();
    let pawns = board.board[board.to_move as usize][Pawn as usize];
    let vacancies = !board.occupancies();
    let non_promotions = match board.to_move {
        Color::White => pawns & !RANK7,
        Color::Black => pawns & !RANK2,
    };
    let promotions = match board.to_move {
        Color::White => pawns & RANK7,
        Color::Black => pawns & RANK2,
    };
    let up = match board.to_move {
        Color::White => Direction::North,
        Color::Black => Direction::South,
    };
    let up_left = match board.to_move {
        Color::White => Direction::NorthWest,
        Color::Black => Direction::SouthEast,
    };
    let up_right = match board.to_move {
        Color::White => Direction::NorthEast,
        Color::Black => Direction::SouthWest,
    };
    let rank3_bb = match board.to_move {
        Color::White => RANK3,
        Color::Black => RANK6,
    };
    let enemies = board.color_occupancies(opposite_color(board.to_move));

    // Single and double pawn pushes w/o captures
    let mut push_one = vacancies & shift(non_promotions, up);
    let mut push_two = vacancies & shift(push_one & rank3_bb, up);
    while push_one > 0 {
        let dest = pop_lsb(&mut push_one) as u8;
        let src = (dest as i8 - up as i8) as u8;
        moves.push(Move::new(src, dest, None, MoveType::Normal));
    }
    while push_two > 0 {
        let dest = pop_lsb(&mut push_two) as u8;
        let src = (dest as i8 - up as i8 - up as i8) as u8;
        moves.push(Move::new(src, dest, None, MoveType::Normal));
    }

    if promotions > 0 {
        // Promotions - captures and straight pushes
        let mut no_capture_promotions = shift(promotions, up) & vacancies;
        let mut left_capture_promotions = shift(promotions, up_left) & enemies;
        let mut right_capture_promotions = shift(promotions, up_right) & enemies;
        while no_capture_promotions > 0 {
            generate_promotions(pop_lsb(&mut no_capture_promotions), up, &mut moves);
        }
        while left_capture_promotions > 0 {
            generate_promotions(pop_lsb(&mut left_capture_promotions), up_left, &mut moves);
        }
        while right_capture_promotions > 0 {
            generate_promotions(pop_lsb(&mut right_capture_promotions), up_right, &mut moves);
        }
    }

    // Captures
    let mut left_captures = shift(non_promotions, up_left) & enemies;
    let mut right_captures = shift(non_promotions, up_right) & enemies;
    while left_captures > 0 {
        let dest = pop_lsb(&mut left_captures) as u8;
        let src = (dest as i8 - up_left as i8) as u8;
        moves.push(Move::new(src, dest, None, MoveType::Normal));
    }
    while right_captures > 0 {
        let dest = pop_lsb(&mut right_captures) as u8;
        let src = (dest as i8 - up_right as i8) as u8;
        moves.push(Move::new(src, dest, None, MoveType::Normal));
    }

    // En Passant
    if board.can_en_passant() {
        let p1 = board.piece_on_square((board.en_passant_square - up_left as i8) as usize);
        let p2 = board.piece_on_square((board.en_passant_square - up_right as i8) as usize);
        if let Some(p1) = p1 {
            if p1 == Pawn {
                let dest = board.en_passant_square as u8;
                let src = (dest as i8 - up_left as i8) as u8;
                moves.push(Move::new(src, dest, None, MoveType::EnPassant));
            }
        }
        if let Some(p2) = p2 {
            if p2 == Pawn {
                let dest = board.en_passant_square as u8;
                let src = (dest as i8 - up_right as i8) as u8;
                moves.push(Move::new(src, dest, None, MoveType::EnPassant));
            }
        }
    }

    moves
}

fn generate_promotions(dest: u64, d: Direction, moves: &mut Vec<Move>) {
    for p in Promotion::iter() {
        moves.push(Move::new(
            (dest as i8 - d as i8) as u8,
            dest as u8,
            Some(p),
            MoveType::Promotion,
        ));
    }
}

fn generate_bitboard_moves(board: &Board, piece_name: PieceName) -> Vec<Move> {
    let mut moves = Vec::new();
    if board.board[board.to_move as usize][piece_name as usize] == 0 {
        return moves;
    }
    for square in 0..63 {
        if board.square_contains_piece(piece_name, board.to_move, square) {
            let enemies = !board.color_occupancies(board.to_move);
            let attack_bitboard = match piece_name {
                PieceName::King => king_attacks(square),
                PieceName::Queen => {
                    get_rook_moves(square, enemies) | get_bishop_moves(square, enemies)
                }
                PieceName::Rook => get_rook_moves(square, enemies),
                PieceName::Bishop => get_bishop_moves(square, enemies),
                PieceName::Knight => knight_attacks(square),
                Pawn => panic!(),
            };
            let attacks = attack_bitboard & enemies;
            push_moves(&mut moves, attacks, square);
        }
    }
    moves
}

fn push_moves(moves: &mut Vec<Move>, mut attacks: u64, square: usize) {
    let mut idx = 0;
    while attacks != 0 {
        if attacks & 1 != 0 {
            moves.push(Move::new(square as u8, idx, None, MoveType::Normal));
        }
        attacks >>= 1;
        idx += 1;
    }
}

/// Filters out moves that are captures for quiescence search
pub fn generate_quiet_moves(board: &Board) -> Vec<Move> {
    let legal_moves = generate_moves(board);
    legal_moves
        .into_iter()
        .filter(|m| bit_is_off(board.occupancies(), m.dest_square().into()))
        .collect::<Vec<Move>>()
}

pub fn generate_moves(board: &Board) -> Vec<Move> {
    let psuedolegal = generate_psuedolegal_moves(board);

    psuedolegal
        .into_iter()
        .filter(|m| {
            let mut new_b = *board;
            new_b.make_move(m);
            !new_b.under_attack(board.to_move)
        })
        .collect::<Vec<Move>>()
}

impl Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut str = String::new();
        str += "Start: ";
        str += &self.origin_square().to_string();
        str += " End: ";
        str += &self.dest_square().to_string();
        str += " Castle: ";
        // match self.castle {
        //     Castle::None => str += "No Castle ",
        //     Castle::WhiteKingCastle => str += "White King castle ",
        //     Castle::WhiteQueenCastle => str += "white queen castle ",
        //     Castle::BlackKingCastle => str += "black king castle ",
        //     Castle::BlackQueenCastle => str += "black queen castle ",
        // }
        str += &self.is_castle().to_string();
        str += " Promotion: ";
        match self.promotion() {
            Some(Promotion::Queen) => str += "Queen ",
            Some(Promotion::Rook) => str += "Rook ",
            Some(Promotion::Bishop) => str += "Bishop ",
            Some(Promotion::Knight) => str += "Knight ",
            None => str += "None ",
        }
        // match self.capture {
        //     None => {
        //         str += " Nothing Captured ";
        //     }
        //     Some(piece_name) => match piece_name {
        //         PieceName::King => str += " Captured a King  ",
        //         PieceName::Queen => str += " Captured a Queen ",
        //         PieceName::Rook => str += " Captured a Rook ",
        //         PieceName::Bishop => str += " Captured a Bishop ",
        //         PieceName::Knight => str += " Captured a Knight ",
        //         PieceName::Pawn => str += " Captured a Pawn ",
        //     },
        // }
        // match self.piece_moving {
        //     PieceName::King => str += " King moving ",
        //     PieceName::Queen => str += " Queen moving ",
        //     PieceName::Bishop => str += " Bishop moving ",
        //     PieceName::Rook => str += " Rook moving ",
        //     PieceName::Knight => str += " Knight moving ",
        //     PieceName::Pawn => str += " Pawn moving ",
        // }
        str += " En Passant: ";
        str += &self.is_en_passant().to_string();
        str += "  ";
        str += &self.to_lan();
        write!(f, "{}", str)
    }
}

#[cfg(test)]
mod move_test {
    use super::*;

    #[test]
    fn test_move_creation() {
        let normal_move = Move::new(10, 20, None, MoveType::Normal);
        assert_eq!(normal_move.origin_square(), 10);
        assert_eq!(normal_move.dest_square(), 20);
        assert!(!normal_move.is_castle());
        assert!(!normal_move.is_en_passant());
        assert_eq!(normal_move.promotion(), None);

        let promotion_move = Move::new(15, 25, Some(Promotion::Queen), MoveType::Promotion);
        assert_eq!(promotion_move.origin_square(), 15);
        assert_eq!(promotion_move.dest_square(), 25);
        assert!(!promotion_move.is_castle());
        assert!(!promotion_move.is_en_passant());
        assert_eq!(promotion_move.promotion(), Some(Promotion::Queen));

        let castle_move = Move::new(4, 2, None, MoveType::Castle);
        assert_eq!(castle_move.origin_square(), 4);
        assert_eq!(castle_move.dest_square(), 2);
        assert!(castle_move.is_castle());
        assert!(!castle_move.is_en_passant());
        assert_eq!(castle_move.promotion(), None);

        let en_passant_move = Move::new(7, 5, None, MoveType::EnPassant);
        assert_eq!(en_passant_move.origin_square(), 7);
        assert_eq!(en_passant_move.dest_square(), 5);
        assert!(!en_passant_move.is_castle());
        assert!(en_passant_move.is_en_passant());
        assert_eq!(en_passant_move.promotion(), None);
    }

    #[test]
    fn test_promotion_conversion() {
        let knight_promotion = Move::new(0, 7, Some(Promotion::Knight), MoveType::Promotion);
        assert_eq!(knight_promotion.promotion(), Some(Promotion::Knight));

        let bishop_promotion = Move::new(15, 23, Some(Promotion::Bishop), MoveType::Promotion);
        assert_eq!(bishop_promotion.promotion(), Some(Promotion::Bishop));

        let rook_promotion = Move::new(28, 31, Some(Promotion::Rook), MoveType::Promotion);
        assert_eq!(rook_promotion.promotion(), Some(Promotion::Rook));

        let queen_promotion = Move::new(62, 61, Some(Promotion::Queen), MoveType::Promotion);
        assert_eq!(queen_promotion.promotion(), Some(Promotion::Queen));
    }
}

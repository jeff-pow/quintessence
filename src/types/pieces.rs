use strum_macros::EnumIter;

use crate::search::pvs::INFINITY;

pub const KING_PTS: i32 = INFINITY;
pub const QUEEN_PTS: i32 = 1000;
pub const ROOK_PTS: i32 = 525;
pub const BISHOP_PTS: i32 = 350;
pub const KNIGHT_PTS: i32 = 350;
pub const PAWN_PTS: i32 = 100;
pub const NUM_PIECES: usize = 6;

#[derive(EnumIter, Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// Returns the opposite color
    pub fn opp(&self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Debug, EnumIter, Copy, Clone, PartialEq, Eq)]
#[repr(usize)]
pub enum PieceName {
    King = 0,
    Queen = 1,
    Rook = 2,
    Bishop = 3,
    Knight = 4,
    Pawn = 5,
}

impl PieceName {
    #[inline(always)]
    pub fn value(&self) -> i32 {
        match self {
            PieceName::King => KING_PTS,
            PieceName::Queen => QUEEN_PTS,
            PieceName::Rook => ROOK_PTS,
            PieceName::Bishop => BISHOP_PTS,
            PieceName::Knight => KNIGHT_PTS,
            PieceName::Pawn => PAWN_PTS,
        }
    }
}

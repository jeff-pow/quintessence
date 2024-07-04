use crate::impl_index;
use std::{
    mem::transmute,
    ops::{self, Index, IndexMut},
};

impl_index!(Color);
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

#[macro_export]
macro_rules! impl_index {
    ($enum_name:ident) => {
        impl<T, const N: usize> Index<$enum_name> for [T; N] {
            type Output = T;

            fn index(&self, index: $enum_name) -> &Self::Output {
                &self[index as usize]
            }
        }

        impl<T, const N: usize> IndexMut<$enum_name> for [T; N] {
            fn index_mut(&mut self, index: $enum_name) -> &mut Self::Output {
                &mut self[index as usize]
            }
        }
    };
}

impl Color {
    pub const fn idx(self) -> usize {
        match self {
            Self::White => 0,
            Self::Black => 1,
        }
    }

    fn from_u32(val: u32) -> Self {
        match val {
            0 => Self::White,
            1 => Self::Black,
            _ => panic!("Unexpected value"),
        }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        [Self::White, Self::Black].into_iter()
    }
}

impl ops::Not for Color {
    type Output = Self;
    fn not(self) -> Self::Output {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

impl From<usize> for Color {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::White,
            1 => Self::Black,
            _ => panic!("Invalid color index"),
        }
    }
}

pub const NUM_PIECES: usize = 6;

impl_index!(PieceName);
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceName {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    None,
}

impl PieceName {
    pub fn value(self) -> i32 {
        match self {
            Self::Pawn => 100,
            Self::Knight => 313,
            Self::Bishop => 310,
            Self::Rook => 514,
            Self::Queen => 1002,
            Self::King => 0,
            Self::None => panic!("Invalid piece"),
        }
    }

    pub(crate) const fn idx(self) -> usize {
        self as usize
    }

    fn from_u32(val: u32) -> Self {
        assert!((0..7).contains(&val));
        unsafe { transmute(val as u8) }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        [Self::Pawn, Self::Knight, Self::Bishop, Self::Rook, Self::Queen, Self::King].into_iter()
    }
}

impl_index!(Piece);
#[derive(Eq, Copy, Clone, PartialEq, Debug, Default)]
#[repr(u8)]
pub enum Piece {
    WhitePawn,
    BlackPawn,

    WhiteKnight,
    BlackKnight,

    WhiteBishop,
    BlackBishop,

    WhiteRook,
    BlackRook,

    WhiteQueen,
    BlackQueen,

    WhiteKing,
    BlackKing,

    #[default]
    None,
}

impl Piece {
    pub(crate) fn new(name: PieceName, color: Color) -> Self {
        unsafe { transmute(((name as u32) << 1 | color as u32) as u8) }
    }

    pub(crate) fn name(self) -> PieceName {
        PieceName::from_u32(self as u32 >> 1)
    }

    pub(crate) fn value(self) -> i32 {
        self.name().value()
    }

    pub(crate) fn color(self) -> Color {
        Color::from_u32(self as u32 & 0b1)
    }

    pub(crate) fn from_u32(x: u32) -> Self {
        assert!((0..13).contains(&x));
        unsafe { transmute(x as u8) }
    }

    pub(crate) fn char(self) -> String {
        let c = match self.name() {
            PieceName::Pawn => "P",
            PieceName::Knight => "N",
            PieceName::Bishop => "B",
            PieceName::Rook => "R",
            PieceName::Queen => "Q",
            PieceName::King => "K",
            PieceName::None => "_",
        };
        if self.color() == Color::Black {
            c.to_ascii_lowercase()
        } else {
            c.to_string()
        }
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        [
            Self::WhitePawn,
            Self::BlackPawn,
            Self::WhiteKnight,
            Self::BlackKnight,
            Self::WhiteBishop,
            Self::BlackBishop,
            Self::WhiteRook,
            Self::BlackRook,
            Self::WhiteQueen,
            Self::BlackQueen,
            Self::WhiteKing,
            Self::BlackKing,
        ]
        .into_iter()
    }
}

#[cfg(test)]
mod piece_tests {
    use super::*;

    #[test]
    fn test_new_piece() {
        assert_eq!(Piece::new(PieceName::Pawn, Color::White), Piece::WhitePawn);
        assert_eq!(Piece::new(PieceName::Knight, Color::White), Piece::WhiteKnight);
        assert_eq!(Piece::new(PieceName::Bishop, Color::White), Piece::WhiteBishop);
        assert_eq!(Piece::new(PieceName::Rook, Color::White), Piece::WhiteRook);
        assert_eq!(Piece::new(PieceName::Queen, Color::White), Piece::WhiteQueen);
        assert_eq!(Piece::new(PieceName::King, Color::White), Piece::WhiteKing);

        assert_eq!(Piece::new(PieceName::Pawn, Color::Black), Piece::BlackPawn);
        assert_eq!(Piece::new(PieceName::Knight, Color::Black), Piece::BlackKnight);
        assert_eq!(Piece::new(PieceName::Bishop, Color::Black), Piece::BlackBishop);
        assert_eq!(Piece::new(PieceName::Rook, Color::Black), Piece::BlackRook);
        assert_eq!(Piece::new(PieceName::Queen, Color::Black), Piece::BlackQueen);
        assert_eq!(Piece::new(PieceName::King, Color::Black), Piece::BlackKing);

        for color in [Color::White, Color::Black] {
            for name in [
                PieceName::Pawn,
                PieceName::Knight,
                PieceName::Bishop,
                PieceName::Rook,
                PieceName::Queen,
                PieceName::King,
            ] {
                let piece = Piece::new(name, color);
                assert_eq!(piece.name(), name);
                assert_eq!(piece.color(), color);
            }
        }
    }

    #[test]
    fn test_piece_name_color() {
        let piece = Piece::new(PieceName::Rook, Color::Black);
        assert_eq!(piece.name(), PieceName::Rook);
        assert_eq!(piece.color(), Color::Black);
    }

    #[test]
    fn test_piece_index_conversion() {
        assert_eq!(PieceName::from_u32(0), PieceName::Pawn);
        assert_eq!(PieceName::from_u32(1), PieceName::Knight);
        assert_eq!(PieceName::from_u32(2), PieceName::Bishop);
        assert_eq!(PieceName::from_u32(3), PieceName::Rook);
        assert_eq!(PieceName::from_u32(4), PieceName::Queen);
        assert_eq!(PieceName::from_u32(5), PieceName::King);
        assert_eq!(PieceName::from_u32(6), PieceName::None);
    }

    #[test]
    fn test_piece_color_conversion() {
        assert_eq!(Color::from_u32(0), Color::White);
        assert_eq!(Color::from_u32(1), Color::Black);
    }
}

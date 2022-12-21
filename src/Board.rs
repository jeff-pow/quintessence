use crate::{Piece, Color, PieceName};

pub struct Board {
    pub board: [Option<Piece>; 64],
    pub black_king_castle: bool,
    pub black_queen_castle: bool,
    pub white_king_castle: bool,
    pub white_queen_castle: bool,
}

impl Board {
    pub fn new() -> Self {
        Board { 
            board: [None; 64],
            black_king_castle: true,
            black_queen_castle: true,
            white_king_castle: true,
            white_queen_castle: true,
        }
    }
    pub fn place_piece(&mut self, piece: Piece, new_idx: u8) {
        let piece_old_idx = piece.current_square;
        self.board[piece_old_idx as usize] = None;
        self.board[new_idx as usize] = Some(piece);
    }
}

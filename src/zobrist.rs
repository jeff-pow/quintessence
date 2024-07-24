use crate::{
    board::Board,
    const_array,
    magics::rand_u64,
    types::pieces::{Color, PieceName},
};

#[derive(Debug, PartialEq, Eq)]
/// Contains hashes for each piece square combination, castling possibility, en passant square, and
/// the side to move for the board.
pub struct Zobrist {
    pub piece: [[u64; 64]; 12],
    pub turn: u64,
    pub castling: [u64; 16],
    // 64 squares plus an invalid square
    // Don't bother figuring out invalid enpassant squares, literally not worth the squeeze
    pub en_passant: [u64; 64],
}

pub const ZOBRIST: Zobrist = {
    let mut prev = 0xE926_E621_0D9E_3487u64;

    let turn_hash = prev;
    prev = rand_u64(prev);

    let piece_square_hashes = const_array!(|p, 12| const_array!(|sq, 64| {
        prev = rand_u64(prev);
        prev
    }));
    let castling = const_array!(|c, 16| {
        prev = rand_u64(prev);
        prev
    });
    let en_passant = const_array!(|sq, 64| {
        prev = rand_u64(prev);
        prev
    });
    Zobrist { piece: piece_square_hashes, turn: turn_hash, castling, en_passant }
};

impl Board {
    /// Provides a hash for the board eval to be placed into a transposition table
    pub(crate) fn generate_hash(&self) -> u64 {
        let mut hash = 0;

        for sq in self.occupancies() {
            let p = self.piece_at(sq);
            hash ^= ZOBRIST.piece[p][sq];
        }

        if let Some(x) = self.en_passant_square {
            hash ^= ZOBRIST.en_passant[x];
        }

        hash ^= ZOBRIST.castling[self.castling_rights as usize];

        if self.stm == Color::Black {
            hash ^= ZOBRIST.turn;
        }

        hash
    }

    pub fn pawn_hash(&self) -> u64 {
        let mut hash = 0;

        for sq in self.piece(PieceName::Pawn) {
            hash ^= ZOBRIST.piece[self.piece_at(sq)][sq];
        }
        // TODO: Test adding stm hash and/or king squares

        hash
    }
}

#[cfg(test)]
mod hashing_test {
    use crate::{board::Board, chess_move::Move, fen};

    #[test]
    fn test_hashing() {
        let board1 = Board::from_fen(fen::STARTING_FEN);
        let board2 = Board::from_fen("4r3/4k3/8/4K3/8/8/8/8 w - - 0 1");
        let board3 = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_ne!(board1.generate_hash(), board2.generate_hash());
        assert_eq!(board1.generate_hash(), board3.generate_hash());
    }

    #[test]
    fn incremental_generation() {
        let board = Board::from_fen("k7/3n4/8/2Q5/4pP2/8/8/K7 b - f3 0 1");
        let mut en_p = board;
        let _ = en_p.make_move(Move::from_san("e4f3", &board));
        assert_eq!(en_p.zobrist_hash, en_p.generate_hash());

        let mut capture = board;
        let _ = capture.make_move(Move::from_san("d7c5", &capture));
        assert_eq!(capture.zobrist_hash, capture.generate_hash());

        let mut quiet = board;
        let _ = quiet.make_move(Move::from_san("a1a2", &quiet));
        assert_eq!(quiet.zobrist_hash, quiet.generate_hash());
    }
}

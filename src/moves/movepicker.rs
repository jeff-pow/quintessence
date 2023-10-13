use crate::engine::perft::count_moves;
use crate::moves::movegenerator::MGT;
use crate::search::killers::{KillerMoves, NUM_KILLER_MOVES};
use crate::{board::board::Board, moves::movegenerator::generate_psuedolegal_moves};

use super::{movelist::MoveList, moves::Move};

#[derive(Default, PartialEq)]
enum MovePickerPhase {
    #[default]
    TTMove,
    CapturesInit,
    Captures,
    KillerMovesInit,
    KillerMoves,
    QuietsInit,
    Quiets,
}

pub struct MovePicker<'a> {
    phase: MovePickerPhase,
    moves: MoveList,
    processed_idx: usize,
    tt_move: Move,
    board: &'a Board,
    killers: [Move; NUM_KILLER_MOVES],
    gen_quiets: bool,
}

impl<'a> Iterator for MovePicker<'a> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        if self.phase == MovePickerPhase::TTMove {
            self.phase = MovePickerPhase::CapturesInit;
            if self.tt_move.is_valid(self.board) {
                return Some(self.tt_move);
            }
        }

        if self.phase == MovePickerPhase::CapturesInit {
            self.phase = MovePickerPhase::Captures;
            self.processed_idx = 0;
            self.moves = generate_psuedolegal_moves(self.board, MGT::All);
            self.moves.score_move_list(self.board, self.tt_move, &self.killers);
        }

        'captures: {
            if self.phase == MovePickerPhase::Captures {
                let m = self.moves.next();
                match m {
                    None => {
                        self.phase = MovePickerPhase::KillerMovesInit;
                        break 'captures;
                    }
                    Some(m) => {
                        assert_ne!(m, Move::NULL);
                        assert!(m.is_valid(self.board));
                        self.processed_idx += 1;
                        return Some(m);
                    }
                }
            }
        }

        if !self.gen_quiets {
            return None;
        }

        if self.phase == MovePickerPhase::KillerMovesInit {
            self.phase = MovePickerPhase::KillerMoves;
            self.moves = MoveList::default();
            self.processed_idx = 0;
            for m in self.killers {
                if m.is_valid(self.board) {
                    self.moves.push(m);
                }
            }
            self.moves.score_move_list(self.board, self.tt_move, &self.killers);
        }

        'killers: {
            if self.phase == MovePickerPhase::KillerMoves {
                let m = self.moves.next();
                match m {
                    None => {
                        self.phase = MovePickerPhase::QuietsInit;
                        break 'killers;
                    }
                    Some(m) => {
                        assert_ne!(m, Move::NULL);
                        assert!(m.is_valid(self.board));
                        self.processed_idx += 1;
                        return Some(m);
                    }
                }
            }
        }

        if self.phase == MovePickerPhase::QuietsInit {
            self.phase = MovePickerPhase::Quiets;
            self.processed_idx = 0;
            self.moves = generate_psuedolegal_moves(self.board, MGT::All);
            self.moves.score_move_list(self.board, self.tt_move, &self.killers);
        }

        if self.phase == MovePickerPhase::Quiets {
            return self.moves.next();
        }

        unreachable!()
    }
}

impl<'a> MovePicker<'a> {
    #[allow(dead_code)]
    fn dummy(board: &'a Board) -> Self {
        MovePicker {
            tt_move: Move::NULL,
            board,
            killers: [Move::NULL; NUM_KILLER_MOVES],
            phase: MovePickerPhase::TTMove,
            moves: MoveList::default(),
            processed_idx: 0,
            gen_quiets: true,
        }
    }

    pub fn qsearch(board: &'a Board, tt_move: Move, gen_quiets: bool) -> Self {
        MovePicker {
            tt_move,
            board,
            killers: [Move::NULL; NUM_KILLER_MOVES],
            phase: MovePickerPhase::TTMove,
            moves: MoveList::default(),
            processed_idx: 0,
            gen_quiets,
        }
    }

    pub fn new(board: &'a Board, ply: i32, tt_move: Move, killers: &KillerMoves) -> Self {
        MovePicker {
            tt_move,
            board,
            killers: killers[ply as usize],
            phase: MovePickerPhase::TTMove,
            moves: MoveList::default(),
            processed_idx: 0,
            gen_quiets: true,
        }
    }
}

#[allow(dead_code)]
fn perft(board: &Board, depth: i32) -> usize {
    let mut total = 0;
    let moves = MovePicker::qsearch(board, Move::NULL, true);
    for m in moves {
        let mut new_b = board.to_owned();
        new_b.make_move(m);
        let count = count_moves(depth - 1, &new_b);
        total += count;
        println!("{}: {}", m.to_lan(), count);
    }
    println!("\nNodes searched: {}", total);
    total
}

#[cfg(test)]
mod move_picker_tests {
    use crate::{
        board::fen::{self, build_board},
        moves::{
            movelist::MoveList,
            movepicker::{perft, MovePicker},
            moves::{from_lan, Move},
        },
        search::killers::NUM_KILLER_MOVES,
    };

    #[test]
    fn test_movegenerator() {
        let board = build_board(fen::STARTING_FEN);
        let gen = MovePicker::dummy(&board);
        let mut count = 0;
        for m in gen {
            assert_ne!(m, Move::NULL);
            assert!(m.is_valid(&board));
            dbg!(m.to_lan());
            count += 1;
        }
        assert_eq!(20, count);
    }

    #[test]
    fn full_generator() {
        let board = build_board("k7/2r3n1/8/3q1b2/4P3/8/3N4/KR6 w - - 0 1");
        let gen = MovePicker {
            phase: super::MovePickerPhase::TTMove,
            moves: MoveList::default(),
            processed_idx: 0,
            tt_move: from_lan("e4d5", &board),
            board: &board,
            killers: [from_lan("b1b5", &board), from_lan("b1c1", &board)],
            gen_quiets: true,
        };
        for m in gen {
            assert!(m.is_valid(&board));
        }
    }

    #[test]
    fn null_moves() {
        let board = build_board("k7/2r3n1/8/3q1b2/8/8/3N4/KR6 w - - 0 1");
        let gen = MovePicker {
            phase: super::MovePickerPhase::TTMove,
            moves: MoveList::default(),
            processed_idx: 0,
            tt_move: Move::NULL,
            board: &board,
            killers: [Move::NULL; NUM_KILLER_MOVES],
            gen_quiets: true,
        };
        for m in gen {
            assert!(m.is_valid(&board));
        }
    }

    #[test]
    fn movepicker_perft() {
        let board = build_board(fen::STARTING_FEN);
        assert_eq!(119_060_324, perft(&board, 6));
    }
}

use super::{
    movelist::{MoveList, MoveListEntry},
    moves::Move,
};
use crate::{board::board::Board, moves::movegenerator::MGT, search::thread::ThreadData};

pub const TT_MOVE_SCORE: i32 = i32::MAX - 1000;
pub const KILLER: i32 = 1_000_000;
pub const COUNTER_MOVE: i32 = 800_000;

#[derive(PartialEq, PartialOrd, Eq)]
pub enum MovePickerPhase {
    TTMove,

    CapturesInit,
    GoodCaptures,

    Killer,
    Counter,

    QuietsInit,
    Quiets,

    BadCaptures,

    Finished,
}

pub struct MovePicker {
    pub phase: MovePickerPhase,
    skip_quiets: bool,
    margin: i32,

    moves: MoveList,

    tt_move: Move,
    killer_move: Move,
    counter_move: Move,

    current: usize,
    end: usize,
    end_bad: usize,
}

impl MovePicker {
    pub fn new(tt_move: Move, td: &ThreadData, margin: i32, skip_quiets: bool) -> Self {
        let prev = td.stack.prev_move(td.ply - 1);
        let counter_move = td.history.get_counter(prev);
        Self {
            moves: MoveList::default(),
            phase: MovePickerPhase::TTMove,
            margin,
            tt_move,
            killer_move: td.stack[td.ply].killer_move,
            counter_move,
            skip_quiets,
            current: 0,
            end: 0,
            end_bad: 0,
        }
    }

    /// Select the next move to try. Returns None if there are no more moves to try.
    pub fn next(&mut self, board: &Board, td: &ThreadData) -> Option<MoveListEntry> {
        if self.phase == MovePickerPhase::TTMove {
            self.phase = MovePickerPhase::CapturesInit;
            if board.occupancies().empty(self.tt_move.to()) && self.skip_quiets {
                return self.next(board, td);
            }
            if board.is_pseudo_legal(self.tt_move) {
                return Some(MoveListEntry { m: self.tt_move, score: TT_MOVE_SCORE });
            }
        }

        if self.phase == MovePickerPhase::CapturesInit {
            self.phase = MovePickerPhase::GoodCaptures;
            board.generate_moves(MGT::CapturesOnly, &mut self.moves);
            score_captures(td, board, &mut self.moves.arr);
            self.end = self.moves.len();
        }

        if self.phase == MovePickerPhase::GoodCaptures {
            if self.current != self.end {
                let entry = self.moves.pick_move(self.current, self.end);
                self.current += 1;

                if entry.m == self.tt_move {
                    return self.next(board, td);
                } else if !board.see(entry.m, self.margin) {
                    self.moves.arr[self.end_bad] = self.moves.arr[self.current - 1];
                    self.end_bad += 1;
                    return self.next(board, td);
                } else {
                    return Some(entry);
                }
            }

            self.phase = if self.skip_quiets {
                self.current = 0;
                self.end = self.end_bad;
                MovePickerPhase::Finished
            } else {
                MovePickerPhase::Killer
            };
        }

        if self.phase == MovePickerPhase::Killer {
            self.phase = MovePickerPhase::Counter;
            if !self.skip_quiets
                && self.killer_move != self.tt_move
                && board.is_pseudo_legal(self.killer_move)
            {
                return Some(MoveListEntry { m: self.killer_move, score: KILLER });
            }
        }

        if self.phase == MovePickerPhase::Counter {
            self.phase = MovePickerPhase::QuietsInit;
            if !self.skip_quiets
                && self.counter_move != self.tt_move
                && self.counter_move != self.killer_move
                && board.is_pseudo_legal(self.counter_move)
            {
                return Some(MoveListEntry { m: self.counter_move, score: COUNTER_MOVE });
            }
        }

        if self.phase == MovePickerPhase::QuietsInit {
            self.phase = MovePickerPhase::Quiets;
            if !self.skip_quiets {
                self.current = self.end_bad;
                let len = self.moves.len();
                board.generate_moves(MGT::QuietsOnly, &mut self.moves);
                self.end = self.moves.len();
                score_quiets(td, &mut self.moves.arr[len..]);
            }
        }

        if self.phase == MovePickerPhase::Quiets {
            if self.current != self.end && !self.skip_quiets {
                let entry = self.moves.pick_move(self.current, self.end);
                self.current += 1;

                if self.is_cached(entry.m) {
                    return self.next(board, td);
                } else {
                    return Some(entry);
                }
            }

            self.current = 0;
            self.end = self.end_bad;
            self.phase = MovePickerPhase::BadCaptures;
        }

        if self.phase == MovePickerPhase::BadCaptures {
            if self.current != self.end {
                // Note: Moves are already sorted by this point, so we don't need to use the
                // mixed select-sort function in MoveList to get the next move, that's just wasting
                // work.
                let entry = self.moves.arr[self.current];
                self.current += 1;
                if entry.m == self.tt_move {
                    return self.next(board, td);
                } else {
                    return Some(entry);
                }
            }
            self.phase = MovePickerPhase::Finished;
        }

        None
    }

    /// Determines if a move is stored as a special move by the move picker
    fn is_cached(&self, m: Move) -> bool {
        m == self.tt_move || m == self.killer_move || m == self.counter_move
    }
}

fn score_quiets(td: &ThreadData, moves: &mut [MoveListEntry]) {
    for MoveListEntry { m, score } in moves {
        *score = td.history.quiet_history(*m, &td.stack, td.ply);
    }
}

fn score_captures(td: &ThreadData, board: &Board, moves: &mut [MoveListEntry]) {
    for MoveListEntry { m, score } in moves {
        *score = td.history.capt_hist(*m, board);
    }
}

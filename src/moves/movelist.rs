use crate::{
    board::board::Board,
    search::{killers::NUM_KILLER_MOVES, see::see, SearchInfo},
    types::pieces::PieceName,
};
use std::{mem::MaybeUninit, ops::Index};

use super::moves::{Move, Promotion};

pub const MAX_LEN: usize = 218;
#[derive(Copy, Clone, Debug)]
/// Movelist elements contains a move and an i32 where a score can be stored later to be used in move ordering
/// for efficient search pruning
pub struct MoveList {
    pub arr: [MoveListEntry; MAX_LEN],
    len: usize,
    current_idx: usize,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MoveListEntry {
    pub m: Move,
    pub score: i32,
}

impl MoveListEntry {
    fn new(m: Move, score: i32) -> Self {
        MoveListEntry { m, score }
    }
}

impl MoveList {
    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_LEN);
        self.arr[self.len] = MoveListEntry::new(m, 0);
        self.len += 1;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    fn swap(&mut self, a: usize, b: usize) {
        unsafe {
            let ptr_a: *mut MoveListEntry = &mut self.arr[a];
            let ptr_b: *mut MoveListEntry = &mut self.arr[b];
            std::ptr::swap(ptr_a, ptr_b);
        }
    }

    #[inline(always)]
    /// Sorts next move into position and then returns a reference to the move
    fn pick_move(&mut self, idx: usize) -> MoveListEntry {
        self.sort_next_move(idx);
        self.arr[idx]
    }

    #[inline(always)]
    pub fn into_vec(self) -> Vec<Move> {
        let mut v = Vec::new();
        self.into_iter().for_each(|x| v.push(x.m));
        v
    }

    #[inline(always)]
    pub fn perft_next(&mut self) -> Option<Move> {
        if self.current_idx >= self.len {
            None
        } else {
            let m = self.arr[self.current_idx];
            self.current_idx += 1;
            Some(m.m)
        }
    }

    fn sort_next_move(&mut self, idx: usize) {
        let mut max_idx = idx;
        for i in (idx + 1)..self.len {
            if self.arr[i].score > self.arr[max_idx].score {
                max_idx = i;
            }
        }
        self.swap(max_idx, idx);
    }

    pub fn score_moves(
        &mut self,
        board: &Board,
        table_move: Move,
        killers: &[Move; NUM_KILLER_MOVES],
        info: &SearchInfo,
    ) {
        for i in 0..self.len {
            let entry = &mut self.arr[i];
            let m = &mut entry.m;
            let score = &mut entry.score;
            let piece_moving = board.piece_at(m.origin_square()).unwrap();
            let capture = board.capture(*m);
            let promotion = m.promotion();
            let prev = info.current_line.last().unwrap_or(&Move::NULL);
            let counter = info.history.get_counter(board.to_move, *prev);
            if *m == table_move {
                *score = TTMOVE;
            } else if let Some(promotion) = promotion {
                match promotion {
                    Promotion::Queen => *score = QUEEN_PROMOTION,
                    _ => *score = BAD_PROMOTION,
                }
            } else if let Some(c) = capture {
                *score = if see(board, *m, -PieceName::Pawn.value()) {
                    GOOD_CAPTURE
                } else {
                    BAD_CAPTURE
                } + MVV_LVA[piece_moving.idx()][c.idx()];
            } else if killers[0] == *m {
                *score = KILLER_ONE;
            } else if killers[1] == *m {
                *score = KILLER_TWO;
            } else if counter == *m {
                *score = COUNTER_MOVE;
            } else {
                *score = info.history.get_history(*m, board.to_move);
            }
        }
    }
}

const QUEEN_PROMOTION: i32 = 20000001;
const GOOD_CAPTURE: i32 = 3000000;
const KILLER_ONE: i32 = 2000000;
const KILLER_TWO: i32 = 1000000;
const COUNTER_MOVE: i32 = 900000;
const BAD_CAPTURE: i32 = -10000;
const BAD_PROMOTION: i32 = -20000001;
const TTMOVE: i32 = i32::MAX - 1000;
/// [Attacker][Victim]
#[rustfmt::skip]
const MVV_LVA: [[i32; 6]; 6] = [
// Victims
//   K   Q   R   B   N   P       Attacker
    [60, 50, 40, 30, 20, 10], // K
    [61, 51, 41, 31, 21, 11], // Q
    [62, 52, 42, 32, 22, 12], // R
    [63, 53, 43, 33, 23, 13], // B
    [64, 54, 44, 34, 24, 14], // N
    [65, 55, 45, 35, 25, 15], // P
];

impl Index<usize> for MoveList {
    type Output = Move;

    fn index(&self, index: usize) -> &Self::Output {
        &self.arr[index].m
    }
}

impl Iterator for MoveList {
    type Item = MoveListEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.len {
            None
        } else {
            let m = self.pick_move(self.current_idx);
            self.current_idx += 1;
            Some(m)
        }
    }
}

impl FromIterator<MoveListEntry> for MoveList {
    fn from_iter<I: IntoIterator<Item = MoveListEntry>>(iter: I) -> Self {
        let mut move_list = MoveList::default();
        for m in iter {
            move_list.push(m.m);
        }
        move_list
    }
}

impl Default for MoveList {
    fn default() -> Self {
        // Uninitialized memory is much faster than initializing it when the important stuff will
        // be written over anyway ;)
        let arr: MaybeUninit<[MoveListEntry; MAX_LEN]> = MaybeUninit::uninit();
        Self {
            arr: unsafe { arr.assume_init() },
            len: 0,
            current_idx: 0,
        }
    }
}

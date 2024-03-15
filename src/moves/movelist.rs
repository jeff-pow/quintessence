use arrayvec::ArrayVec;

use std::ops::Index;

use super::moves::Move;

pub const MAX_LEN: usize = 218;
#[derive(Clone, Debug, Default)]
/// Movelist elements contains a move and an i32 where a score can be stored later to be used in move ordering
/// for efficient search pruning
pub struct MoveList {
    pub arr: ArrayVec<MoveListEntry, MAX_LEN>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MoveListEntry {
    pub m: Move,
    pub score: i32,
}

impl MoveListEntry {
    const fn new(m: Move, score: i32) -> Self {
        Self { m, score }
    }
}

impl MoveList {
    pub fn push(&mut self, m: Move) {
        self.arr.push(MoveListEntry::new(m, 0));
    }

    pub fn raw_push(&mut self, m: MoveListEntry) {
        self.arr.push(m);
    }

    pub const fn len(&self) -> usize {
        self.arr.len()
    }

    /// Sorts next move into position via partial insertion sort and then returns the move's entry
    pub(super) fn pick_move(&mut self, idx: usize, end: usize) -> MoveListEntry {
        self.sort_next_move(idx, end);
        self.arr[idx]
    }

    fn sort_next_move(&mut self, idx: usize, end: usize) {
        let mut max_idx = idx;
        for i in (idx + 1)..end {
            if self.arr[i].score > self.arr[max_idx].score {
                max_idx = i;
            }
        }
        self.arr.swap(max_idx, idx);
    }
}

impl Index<usize> for MoveList {
    type Output = Move;

    fn index(&self, index: usize) -> &Self::Output {
        &self.arr[index].m
    }
}

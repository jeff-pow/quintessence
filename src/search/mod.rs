use arrayvec::ArrayVec;
use std::ops::{Index, IndexMut};

use self::{game_time::Clock, search::MAX_SEARCH_DEPTH};
use crate::{chess_move::Move, types::pieces::Piece};

pub mod game_time;
pub mod lmr_table;
pub mod quiescence;
pub mod search;

#[derive(Clone, Copy, Default)]
pub struct PlyEntry {
    pub in_check: bool,
    pub capture: Piece,
    pub killer_move: Option<Move>,
    pub played_move: Option<Move>,
    pub static_eval: i32,
    pub cutoffs: u32,
    pub singular: Option<Move>,
    /// Double extensions
    pub multi_extns: i32,
}

#[derive(Default)]
pub struct PV {
    pub line: ArrayVec<Option<Move>, { MAX_SEARCH_DEPTH as usize }>,
}

impl PV {
    fn update(&mut self, m: Option<Move>, other: Self) {
        self.line.clear();
        self.line.push(m);
        self.line.extend(other.line);
    }
}

#[derive(Clone)]
pub struct SearchStack {
    stack: [PlyEntry; MAX_SEARCH_DEPTH as usize + 5],
}

impl SearchStack {
    pub fn prev_move(&self, ply: i32) -> Option<Move> {
        self.stack.get(ply as usize).map(|e| e.played_move)?
    }
}

impl Default for SearchStack {
    fn default() -> Self {
        Self { stack: [PlyEntry::default(); MAX_SEARCH_DEPTH as usize + 5] }
    }
}

impl Index<i32> for SearchStack {
    type Output = PlyEntry;

    fn index(&self, index: i32) -> &Self::Output {
        &self.stack[index as usize]
    }
}

impl IndexMut<i32> for SearchStack {
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        &mut self.stack[index as usize]
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchType {
    /// User has requested a search until a particular depth
    Depth(i32),
    /// Search determines how much time to allow itself
    Time(Clock),
    /// Only search for N nodes
    Nodes(u64),
    /// Search for a mate at the provided depth
    Mate(i32),
    #[default]
    /// Search forever
    Infinite,
}

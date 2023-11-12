use crate::{moves::moves::Move, types::pieces::Color};

pub const MAX_HIST_VAL: i32 = i16::MAX as i32;

#[derive(Clone, Copy)]
pub struct HistoryEntry {
    score: i32,
    counter: Move,
    // King can't be captured, so it doesn't need a square
    capthist: [i32; 5],
}

impl Default for HistoryEntry {
    fn default() -> Self {
        Self {
            score: Default::default(),
            counter: Default::default(),
            capthist: [0; 5],
        }
    }
}

#[derive(Clone)]
pub struct MoveHistory {
    search_history: Box<[[[HistoryEntry; 64]; 6]; 2]>,
}

impl MoveHistory {
    pub fn update_history(&mut self, m: Move, bonus: i32, side: Color) {
        let i = &mut self.search_history[side][m.piece_moving()][m.dest_square()].score;
        *i += bonus - *i * bonus.abs() / MAX_HIST_VAL;
    }

    pub fn get_history(&self, m: Move, side: Color) -> i32 {
        self.get_search_history(m, side)
    }

    fn get_search_history(&self, m: Move, side: Color) -> i32 {
        self.search_history[side][m.piece_moving()][m.dest_square()].score
    }

    pub fn set_counter(&mut self, side: Color, prev: Move, m: Move) {
        self.search_history[side][prev.piece_moving()][prev.dest_square()].counter = m;
    }

    pub fn get_counter(&self, side: Color, m: Move) -> Move {
        if m == Move::NULL {
            Move::NULL
        } else {
            self.search_history[side][m.piece_moving()][m.dest_square()].counter
        }
    }
}

impl Default for MoveHistory {
    fn default() -> Self {
        Self {
            search_history: Box::new([[[HistoryEntry::default(); 64]; 6]; 2]),
        }
    }
}

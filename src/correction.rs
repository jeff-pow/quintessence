use crate::{search::search::NEAR_CHECKMATE, types::pieces::Color};

const NUM_ENTRIES: usize = 16384;
const CORRECTION_GRAIN: i32 = 256;
const WEIGHT_SCALE: i32 = 256;
const CORRECTION_MAX: i32 = CORRECTION_GRAIN * 32;

#[derive(Clone)]
pub struct CorrectionHistory {
    table: [[i32; NUM_ENTRIES]; 2],
}

impl CorrectionHistory {
    pub fn correct_score(&self, stm: Color, pawn_hash: u64, raw_eval: i32) -> i32 {
        (raw_eval + self.table[stm][pawn_hash as usize % NUM_ENTRIES] / CORRECTION_GRAIN)
            .clamp(-NEAR_CHECKMATE + 1, NEAR_CHECKMATE - 1)
    }

    pub fn update_table(&mut self, stm: Color, pawn_hash: u64, depth: i32, diff: i32) {
        let entry = &mut self.table[stm][pawn_hash as usize % NUM_ENTRIES];
        let new_weight = (16).min(depth + 1);
        let scaled_diff = diff * CORRECTION_GRAIN;
        assert!(new_weight <= WEIGHT_SCALE);

        let update = *entry * (WEIGHT_SCALE - new_weight) + scaled_diff * new_weight;
        *entry = (update / WEIGHT_SCALE).clamp(-CORRECTION_MAX, CORRECTION_MAX);
    }
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        Self { table: [[0; NUM_ENTRIES]; 2] }
    }
}

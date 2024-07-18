use super::{Align64, Block, INPUT_SIZE};

use crate::types::{
    pieces::{Color, Piece, NUM_PIECES},
    square::{Square, NUM_SQUARES},
};
use bytemuck::{Pod, Zeroable};
/**
* When changing activation functions, both the normalization factor and QA may need to change
* alongside changing the crelu calls to screlu in simd and serial code.
*/
const QA: f32 = 1.; // CHANGES WITH NET QUANZIZATION
pub(super) const RELU_MIN: f32 = 0f32;
pub(super) const RELU_MAX: f32 = QA;

pub(super) const SCALE: f32 = 400.;

pub const NUM_BUCKETS: usize = 1;

#[rustfmt::skip]
pub static BUCKETS: [usize; 64] = [
    0, 1, 2, 3, 12, 11, 10, 9,
    4, 4, 5, 5, 14, 14, 13, 13,
    6, 6, 6, 6, 15, 15, 15, 15,
    7, 7, 7, 7, 16, 16, 16, 16,
    8, 8, 8, 8, 17, 17, 17, 17,
    8, 8, 8, 8, 17, 17, 17, 17,
    8, 8, 8, 8, 17, 17, 17, 17,
    8, 8, 8, 8, 17, 17, 17, 17,
];

#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct Network {
    pub feature_weights: [Align64<Block>; INPUT_SIZE * NUM_BUCKETS],
    pub feature_bias: Align64<Block>,
    pub output_weights: [Align64<Block>; 2],
    pub output_bias: f32,
}

impl Network {
    pub fn feature_idx(piece: Piece, sq: Square, mut _king: Square, view: Color) -> usize {
        const COLOR_OFFSET: usize = NUM_SQUARES * NUM_PIECES;
        const PIECE_OFFSET: usize = NUM_SQUARES;
        match view {
            Color::White => piece.color().idx() * COLOR_OFFSET + piece.name().idx() * PIECE_OFFSET + sq.idx(),
            Color::Black => {
                (!piece.color()).idx() * COLOR_OFFSET + piece.name().idx() * PIECE_OFFSET + sq.flip_vertical().idx()
            }
        }
    }

    pub fn bucket(view: Color, mut sq: Square) -> usize {
        if view == Color::Black {
            sq = sq.flip_vertical();
        }
        BUCKETS[sq]
    }
}

fn screlu(i: f32) -> f32 {
    crelu(i) * crelu(i)
}

fn crelu(i: f32) -> f32 {
    i.clamp(RELU_MIN, RELU_MAX)
}

pub(super) fn flatten(acc: &Block, weights: &Block) -> f32 {
    acc.iter().zip(weights).map(|(&i, &w)| screlu(i) * (w)).sum()
}

#[cfg(test)]
mod nnue_tests {
    use std::{hint::black_box, time::Instant};

    use crate::{board::Board, fen::STARTING_FEN};

    #[test]
    fn inference_benchmark() {
        let board = Board::from_fen(STARTING_FEN);
        let acc = board.new_accumulator();
        let start = Instant::now();
        let iters = 10_000_000_u128;
        for _ in 0..iters {
            black_box(acc.scaled_evaluate(&board));
        }
        let duration = start.elapsed().as_nanos();
        println!("{} ns per iter", duration / iters);
        dbg!(duration / iters);
    }
}

use super::{
    simd2::{Vec16, Vec32},
    Align64, Block, HIDDEN_SIZE, INPUT_SIZE,
};

use crate::types::{
    pieces::{Color, Piece, NUM_PIECES},
    square::{Square, NUM_SQUARES},
};
/**
* When changing activation functions, both the normalization factor and QA may need to change
* alongside changing the crelu calls to screlu in simd and serial code.
*/
const QA: i32 = 255; // CHANGES WITH NET QUANZIZATION
const QB: i32 = 64;
pub(super) const QAB: i32 = QA * QB;
pub(super) const NORMALIZATION_FACTOR: i32 = QA; // CHANGES WITH SCRELU/CRELU ACTIVATION
pub(super) const RELU_MIN: i16 = 0;
pub(super) const RELU_MAX: i16 = QA as i16;

pub(super) const SCALE: i32 = 400;

pub const NUM_BUCKETS: usize = 9;

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

#[derive(Debug)]
#[repr(C, align(64))]
pub(super) struct Network {
    pub feature_weights: [Align64<Block>; INPUT_SIZE * NUM_BUCKETS],
    pub feature_bias: Align64<Block>,
    pub output_weights: [Align64<Block>; 2],
    pub output_bias: i16,
}

impl Network {
    pub fn feature_idx(piece: Piece, mut sq: Square, mut king: Square, view: Color) -> usize {
        const COLOR_OFFSET: usize = NUM_SQUARES * NUM_PIECES;
        const PIECE_OFFSET: usize = NUM_SQUARES;
        if king.file() > 3 {
            king = king.flip_horizontal();
            sq = sq.flip_horizontal();
        }
        match view {
            Color::White => {
                BUCKETS[king] * INPUT_SIZE
                    + piece.color().idx() * COLOR_OFFSET
                    + piece.name().idx() * PIECE_OFFSET
                    + sq.idx()
            }
            Color::Black => {
                BUCKETS[king.flip_vertical()] * INPUT_SIZE
                    + (!piece.color()).idx() * COLOR_OFFSET
                    + piece.name().idx() * PIECE_OFFSET
                    + sq.flip_vertical().idx()
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

#[cfg(all(not(target_feature = "avx2"), not(feature = "avx512")))]
fn screlu(i: i16) -> i32 {
    crelu(i) * crelu(i)
}

#[cfg(all(not(target_feature = "avx2"), not(feature = "avx512")))]
fn crelu(i: i16) -> i32 {
    i32::from(i.clamp(RELU_MIN, RELU_MAX))
}

pub(super) fn flatten(acc: &Block, weights: &Block) -> i32 {
    let mut sum = Vec32::zero();
    for i in 0..HIDDEN_SIZE / Vec16::POPULATION {
        let us_vector = Vec16::load(acc, i * Vec16::POPULATION);
        let weights = Vec16::load(weights, i * Vec16::POPULATION);
        let crelu_result = us_vector.clipped_relu();
        let v = crelu_result.mul_lo(weights);
        let mul = v.mul_accumulate(crelu_result);
        sum = sum.add(mul);
    }
    sum.reduce_add()
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

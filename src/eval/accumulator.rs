use crate::{
    board::Board,
    chess_move::{Direction, Move},
    eval::{simd::avx2, HIDDEN_SIZE},
    search::search::{MAX_SEARCH_DEPTH, NEAR_CHECKMATE},
    types::{
        bitboard::Bitboard,
        pieces::{Color, Piece, PieceName},
    },
};

use super::{
    network::{flatten, Network, BUCKETS, NORMALIZATION_FACTOR, NUM_BUCKETS, QAB, SCALE},
    Block, NET,
};
use arrayvec::ArrayVec;
use std::{
    arch::x86_64::{
        __m256, __m256i, _mm256_add_epi16, _mm256_load_si256, _mm256_set1_epi16, _mm256_store_pd, _mm256_store_si256,
        _mm256_sub_epi16,
    },
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Accumulator {
    pub capture: Piece,
    pub correct: [bool; 2],
    pub m: Move,
    pub vals: [Block; 2],
}

impl Default for Accumulator {
    fn default() -> Self {
        Self { vals: [NET.feature_bias; 2], correct: [true; 2], m: Move::NULL, capture: Piece::None }
    }
}

impl Index<Color> for Accumulator {
    type Output = Block;

    fn index(&self, index: Color) -> &Self::Output {
        &self.vals[index.idx()]
    }
}

impl IndexMut<Color> for Accumulator {
    fn index_mut(&mut self, index: Color) -> &mut Self::Output {
        &mut self.vals[index.idx()]
    }
}

impl Accumulator {
    pub fn raw_evaluate(&self, stm: Color) -> i32 {
        let (us, them) = (&self[stm], &self[!stm]);
        let weights = &NET.output_weights;
        let output = flatten(us, &weights[0]) + flatten(them, &weights[1]);
        ((i32::from(NET.output_bias) + output / NORMALIZATION_FACTOR) * SCALE / QAB)
            .clamp(-NEAR_CHECKMATE + 1, NEAR_CHECKMATE - 1)
    }

    /// Credit to viridithas for these values and concepts
    pub fn scaled_evaluate(&self, board: &Board) -> i32 {
        let raw = self.raw_evaluate(board.stm);
        let eval = raw * board.mat_scale() / 1024;
        let eval = eval * (200 - board.half_moves as i32) / 200;
        (eval).clamp(-NEAR_CHECKMATE, NEAR_CHECKMATE)
    }

    fn add_sub(&mut self, old: &Accumulator, a1: usize, s1: usize, side: Color) {
        #[cfg(target_feature = "avx2")]
        unsafe {
            avx2::add_sub(&mut self[side], &old[side], a1, s1)
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            let weights = &NET.feature_weights;
            self[side].iter_mut().zip(&weights[a1].0).zip(&weights[s1].0).zip(old[side].iter()).for_each(
                |(((i, &a), &s), &o)| {
                    *i = o + a - s;
                },
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_sub_sub(&mut self, old: &Accumulator, a1: usize, s1: usize, s2: usize, side: Color) {
        #[cfg(target_feature = "avx2")]
        unsafe {
            avx2::add_sub_sub(&mut self[side], &old[side], a1, s1, s2)
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            let weights = &NET.feature_weights;
            self[side]
                .iter_mut()
                .zip(&weights[a1].0)
                .zip(&weights[s1].0)
                .zip(&weights[s2].0)
                .zip(old[side].iter())
                .for_each(|((((i, &a), &s1), &s2), &o)| {
                    *i = o + a - s1 - s2;
                });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_add_sub_sub(&mut self, old: &Accumulator, a1: usize, a2: usize, s1: usize, s2: usize, side: Color) {
        #[cfg(target_feature = "avx2")]
        unsafe {
            avx2::add_add_sub_sub(&mut self[side], &old[side], a1, a2, s1, s2)
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            let weights = &NET.feature_weights;
            self[side]
                .iter_mut()
                .zip(&weights[a1].0)
                .zip(&weights[a2].0)
                .zip(&weights[s1].0)
                .zip(&weights[s2].0)
                .zip(old[side].iter())
                .for_each(|(((((i, &a1), &a2), &s1), &s2), &o)| {
                    *i = o + a1 + a2 - s1 - s2;
                });
        }
    }

    pub(crate) fn lazy_update(&mut self, old: &Accumulator, side: Color, board: &Board) {
        let m = self.m;
        let from = if side == Color::Black { m.from().flip_vertical() } else { m.from() };
        let to = if side == Color::Black { m.to().flip_vertical() } else { m.to() };
        assert!(
            m.piece_moving().name() != PieceName::King
                || m.piece_moving().color() != side
                || BUCKETS[from] == BUCKETS[to]
        );
        let piece_moving = m.promotion().unwrap_or(m.piece_moving());
        let king = board.king_square(side);
        let a1 = Network::feature_idx(piece_moving, m.to(), king, side);
        let s1 = Network::feature_idx(m.piece_moving(), m.from(), king, side);
        if m.is_castle() {
            let rook = Piece::new(PieceName::Rook, m.piece_moving().color());
            let a2 = Network::feature_idx(rook, m.castle_type().rook_to(), king, side);
            let s2 = Network::feature_idx(rook, m.castle_type().rook_from(), king, side);

            self.add_add_sub_sub(old, a1, a2, s1, s2, side)
        } else if self.capture != Piece::None || m.is_en_passant() {
            let cap_square = if m.is_en_passant() {
                match m.piece_moving().color() {
                    Color::White => m.to().shift(Direction::South),
                    Color::Black => m.to().shift(Direction::North),
                }
            } else {
                m.to()
            };
            let capture =
                if m.is_en_passant() { Piece::new(PieceName::Pawn, !m.piece_moving().color()) } else { self.capture };
            let s2 = Network::feature_idx(capture, cap_square, king, side);
            self.add_sub_sub(old, a1, s1, s2, side)
        } else {
            self.add_sub(old, a1, s1, side)
        }
    }
}

// Credit to akimbo. This function streamlines the assembly generated and prevents unnecessary
// redundant loads and stores to the same simd vectors.
pub fn update(block: &mut Block, adds: &[u16], subs: &[u16]) {
    #[cfg(target_feature = "avx2")]
    unsafe {
        avx2::update(block, adds, subs);
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        const REGISTERS: usize = 8;
        const ELEMENTS_PER_LOOP: usize = REGISTERS * 16; // Assumes 16 u16's per 256 bit avx2 register

        let mut regs = [0i16; ELEMENTS_PER_LOOP];

        for i in 0..HIDDEN_SIZE / ELEMENTS_PER_LOOP {
            let offset = ELEMENTS_PER_LOOP * i;

            for (reg, &j) in regs.iter_mut().zip(block[offset..].iter()) {
                *reg = j;
            }

            for &add in adds {
                let weights = &NET.feature_weights[usize::from(add)];

                for (reg, &w) in regs.iter_mut().zip(weights[offset..].iter()) {
                    *reg += w;
                }
            }

            for &sub in subs {
                let weights = &NET.feature_weights[usize::from(sub)];

                for (reg, &w) in regs.iter_mut().zip(weights[offset..].iter()) {
                    *reg -= w;
                }
            }

            for (a, &r) in block[offset..].iter_mut().zip(regs.iter()) {
                *a = r;
            }
        }
    }
}

impl Board {
    pub fn new_accumulator(&self) -> Accumulator {
        let mut acc = Accumulator::default();
        for view in Color::iter() {
            acc.vals[view] = NET.feature_bias;
            let mut vec: ArrayVec<u16, 32> = ArrayVec::new();
            for sq in self.occupancies() {
                let p = self.piece_at(sq);
                let idx = Network::feature_idx(p, sq, self.king_square(view), view);
                vec.push(idx as u16);
            }
            update(&mut acc.vals[view], &vec, &[]);
        }
        acc
    }
}

#[derive(Clone, Debug)]
pub struct AccumulatorStack {
    pub(crate) stack: Vec<Accumulator>,
    /// Top points to the active accumulator, not the space above it
    pub top: usize,
    acc_cache: AccumulatorCache,
}

impl AccumulatorStack {
    pub fn update_stack(&mut self, m: Move, capture: Piece) {
        self.top += 1;
        self.stack[self.top].m = m;
        self.stack[self.top].capture = capture;
        self.stack[self.top].correct = [false; 2];
    }

    fn all_lazy_updates(&mut self, board: &Board, side: Color) {
        let mut curr = self.top;
        while !self.stack[curr].correct[side] {
            curr -= 1;
        }

        while curr < self.top {
            let (bottom, top) = self.stack.split_at_mut(curr + 1);
            top[0].lazy_update(bottom.last().unwrap(), side, board);
            top[0].correct[side] = true;
            curr += 1;
        }
    }

    fn force_updates(&mut self, board: &Board) {
        for color in Color::iter() {
            if !self.stack[self.top].correct[color] {
                if self.can_efficiently_update(color) {
                    self.all_lazy_updates(board, color)
                } else {
                    self.acc_cache.update_acc(board, &mut self.stack[self.top], color);
                    self.stack[self.top].correct[color] = true;
                }
            }
        }
    }

    fn can_efficiently_update(&mut self, side: Color) -> bool {
        let mut curr = self.top;
        loop {
            let m = self.stack[curr].m;
            let from = if side == Color::Black { m.from().flip_vertical() } else { m.from() };
            let to = if side == Color::Black { m.to().flip_vertical() } else { m.to() };

            if m.piece_moving().color() == side
                && m.piece_moving().name() == PieceName::King
                && BUCKETS[from] != BUCKETS[to]
            {
                return false;
            }

            if self.stack[curr].correct[side.idx()] {
                return true;
            }

            curr -= 1;
        }
    }

    pub fn evaluate(&mut self, board: &Board) -> i32 {
        self.force_updates(board);
        assert_eq!(self.stack[self.top].correct, [true; 2]);
        self.top().scaled_evaluate(board)
    }

    pub fn top(&mut self) -> &mut Accumulator {
        &mut self.stack[self.top]
    }

    pub fn pop(&mut self) -> Accumulator {
        self.top -= 1;
        self.stack[self.top + 1]
    }

    pub fn push(&mut self, acc: Accumulator) {
        self.top += 1;
        self.stack[self.top] = acc;
    }

    pub fn clear(&mut self, base_accumulator: &Accumulator) {
        self.stack[0] = *base_accumulator;
        self.top = 0;
    }

    pub fn new(base_accumulator: &Accumulator) -> Self {
        let mut vec = vec![Accumulator::default(); MAX_SEARCH_DEPTH as usize + 50];
        vec[0] = *base_accumulator;
        Self { stack: vec, top: 0, acc_cache: AccumulatorCache::default() }
    }
}

#[derive(Copy, Clone, Debug)]
struct TableEntry {
    acc: Block,
    pieces: [Bitboard; 6],
    color: [Bitboard; 2],
}

impl Default for TableEntry {
    fn default() -> Self {
        Self { acc: NET.feature_bias, pieces: [Bitboard::EMPTY; 6], color: [Bitboard::EMPTY; 2] }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AccumulatorCache {
    entries: Box<[[TableEntry; 2]; NUM_BUCKETS * 2]>,
}

impl AccumulatorCache {
    pub fn update_acc(&mut self, board: &Board, acc: &mut Accumulator, view: Color) {
        let mut adds = ArrayVec::<_, 32>::new();
        let mut subs = ArrayVec::<_, 32>::new();
        let king = board.king_square(view);
        let entry = &mut self.entries[Network::bucket(view, king)][view];

        for piece in Piece::iter() {
            let prev = entry.pieces[piece.name()] & entry.color[piece.color()];
            let curr = board.piece_color(piece.color(), piece.name());

            let added = curr & !prev;
            let removed = prev & !curr;

            for sq in added {
                adds.push(Network::feature_idx(piece, sq, king, view) as u16);
            }
            for sq in removed {
                subs.push(Network::feature_idx(piece, sq, king, view) as u16);
            }
        }

        entry.pieces = board.piece_bbs();
        entry.color = board.color_bbs();

        update(&mut entry.acc, &adds, &subs);
        // unsafe { avx2_update(&mut entry.acc, &adds, &subs) };
        acc.vals[view] = entry.acc;
    }
}

#[cfg(test)]
mod acc_test {
    use super::AccumulatorStack;
    use crate::{board::Board, chess_move::Move};

    macro_rules! make_move_nnue {
        ($board:ident, $stack:ident, $mv_str:literal) => {{
            let m = Move::from_san($mv_str, &$board);
            $stack.update_stack(m, $board.capture(m));
            assert!($board.make_move(m));
        }};
    }

    macro_rules! assert_correct {
        ($board:ident, $stack:ident) => {
            $stack.evaluate(&$board);
            assert_eq!($stack.top().vals, $board.new_accumulator().vals);
        };
    }

    #[test]
    fn lazy_updates() {
        let mut board = Board::from_fen("r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14");
        let mut stack = AccumulatorStack::new(&board.new_accumulator());
        make_move_nnue!(board, stack, "e1g1");

        make_move_nnue!(board, stack, "e8d8");
        assert_correct!(board, stack);
    }

    #[test]
    fn deeper_error() {
        let mut board = Board::from_fen("8/8/1p2k1p1/3p3p/1p1P1P1P/1P2PK2/8/8 w - - 3 54");
        let mut stack = AccumulatorStack::new(&board.new_accumulator());

        make_move_nnue!(board, stack, "e3e4");
        make_move_nnue!(board, stack, "e6e7");
        make_move_nnue!(board, stack, "e4e5");
        make_move_nnue!(board, stack, "e7e6");
        make_move_nnue!(board, stack, "f3g3");
        make_move_nnue!(board, stack, "e6f5");
        make_move_nnue!(board, stack, "g3h3");
        make_move_nnue!(board, stack, "f5f4");
        make_move_nnue!(board, stack, "e5e6");
        make_move_nnue!(board, stack, "f4e3");
        assert_correct!(board, stack);
    }
}

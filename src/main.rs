#![allow(clippy::module_inception)]
#![cfg_attr(feature = "avx512", feature(stdarch_x86_avx512))]

mod attack_boards;
mod bench;
mod board;
mod chess_move;
mod correction;
mod eval;
mod fen;
mod history_table;
mod magics;
mod movegen;
mod movelist;
mod movepicker;
mod perft;
mod search;
mod see;
mod thread;
mod transposition;
mod types;
mod uci;
mod zobrist;

use crate::bench::bench;
use std::env;
use uci::main_loop;

fn main() {
    if env::args().any(|x| x == *"bench") {
        bench();
    } else {
        main_loop();
    }
}

use std::{
    io,
    process::exit,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    thread,
    time::Instant,
};

use crate::{
    board::board::Board,
    engine::{transposition::TranspositionTable, uci::parse_time},
    eval::accumulator::Accumulator,
    moves::moves::Move,
    search::lmr_table::LmrTable,
    search::search::{CHECKMATE, NEAR_CHECKMATE},
};

use super::{
    game_time::Clock,
    history_table::HistoryTable,
    search::{search, MAX_SEARCH_DEPTH},
    AccumulatorStack, SearchStack, SearchType, PV,
};

#[derive(Clone)]
pub struct ThreadData<'a> {
    pub ply: i32,
    // pub max_depth: i32,
    pub iter_max_depth: i32,
    /// Max depth reached by search (include qsearch)
    pub sel_depth: i32,
    pub best_move: Move,

    pub nodes_table: [[u64; 64]; 64],
    pub nodes: AtomicCounter<'a>,
    pub stack: SearchStack,
    pub history: HistoryTable,
    pub hash_history: Vec<u64>,
    pub accumulators: AccumulatorStack,

    // pub game_time: GameTime,
    pub search_start: Instant,
    pub thread_idx: usize,
    pub search_type: SearchType,
    pub halt: &'a AtomicBool,
    pub lmr: &'a LmrTable,
}

impl<'a> ThreadData<'a> {
    pub(crate) fn new(
        halt: &'a AtomicBool,
        hash_history: Vec<u64>,
        thread_idx: usize,
        consts: &'a LmrTable,
        global_nodes: &'a AtomicU64,
    ) -> Self {
        Self {
            ply: 0,
            iter_max_depth: 0,
            stack: SearchStack::default(),
            sel_depth: 0,
            best_move: Move::NULL,
            nodes: AtomicCounter::new(global_nodes),
            history: HistoryTable::default(),
            nodes_table: [[0; 64]; 64],
            accumulators: AccumulatorStack::new(&Accumulator::default()),
            halt,
            search_type: SearchType::default(),
            hash_history,
            thread_idx,
            lmr: consts,
            search_start: Instant::now(),
        }
    }

    pub(super) fn node_tm_stop(&mut self, game_time: Clock, depth: i32) -> bool {
        assert_eq!(0, self.thread_idx);
        let m = self.best_move;
        let frac = self.nodes_table[m.from()][m.to()] as f64 / self.nodes.global_count() as f64;
        let time_scale = if depth > 8 { (1.5 - frac) * 1.4 } else { 0.9 };
        if self.search_start.elapsed().as_millis() as f64
            >= game_time.rec_time.as_millis() as f64 * time_scale
        {
            self.halt.store(true, Ordering::Relaxed);
            return true;
        }
        false
    }

    pub(super) fn soft_stop(&mut self, depth: i32, prev_score: i32) -> bool {
        match self.search_type {
            SearchType::Depth(d) => depth >= d,
            SearchType::Time(time) => {
                self.node_tm_stop(time, depth) || time.soft_termination(self.search_start)
            }
            SearchType::Nodes(n) => self.nodes.global_count() >= n,
            SearchType::Infinite => self.halt.load(Ordering::Relaxed),
            SearchType::Mate(d) => {
                let dist = if prev_score.is_positive() {
                    (CHECKMATE - prev_score + 1) / 2
                } else {
                    -(CHECKMATE + prev_score) / 2
                };
                dist.abs() <= d.abs() || depth > MAX_SEARCH_DEPTH
            }
        }
    }

    pub(super) fn hard_stop(&mut self) -> bool {
        match self.search_type {
            SearchType::Mate(_) | SearchType::Depth(_) | SearchType::Infinite => {
                self.halt.load(Ordering::Relaxed)
            }
            SearchType::Time(time) => time.hard_termination(self.search_start),
            SearchType::Nodes(n) => self.nodes.global_count() >= n,
        }
    }

    pub(super) fn print_search_stats(&self, eval: i32, pv: &PV, tt: &TranspositionTable) {
        let nodes = self.nodes.global_count();
        print!(
            "info time {} depth {} seldepth {} nodes {} nps {} score ",
            self.search_start.elapsed().as_millis(),
            self.iter_max_depth,
            self.sel_depth,
            nodes,
            (nodes as f64 / self.search_start.elapsed().as_secs_f64()) as i64,
        );

        let score = eval;

        if score.abs() >= NEAR_CHECKMATE {
            if score.is_positive() {
                print!("mate {}", (CHECKMATE - score + 1) / 2);
            } else {
                print!("mate {}", (-(CHECKMATE + score) / 2));
            }
        } else {
            print!("cp {score}");
        }

        print!(" hashfull {} pv ", tt.permille_usage());

        for m in pv.line.iter().take(pv.line.len()) {
            print!("{} ", m.to_san());
        }
        println!();
    }

    pub(super) fn is_repetition(&self, board: &Board) -> bool {
        if self.hash_history.len() < 6 {
            return false;
        }

        let mut reps = 2;
        for &hash in self.hash_history.iter().rev().take(board.half_moves + 1) {
            reps -= u32::from(hash == board.zobrist_hash);
            if reps == 0 {
                return true;
            }
        }
        false
    }
}

pub struct ThreadPool<'a> {
    pub main_thread: ThreadData<'a>,
    pub workers: Vec<ThreadData<'a>>,
    pub halt: &'a AtomicBool,
    pub searching: AtomicBool,
}

impl<'a> ThreadPool<'a> {
    pub fn new(
        halt: &'a AtomicBool,
        hash_history: Vec<u64>,
        consts: &'a LmrTable,
        global_nodes: &'a AtomicU64,
    ) -> Self {
        Self {
            main_thread: ThreadData::new(halt, hash_history, 0, consts, global_nodes),
            workers: Vec::new(),
            halt,
            searching: AtomicBool::new(false),
        }
    }

    pub fn reset(&mut self) {
        self.main_thread.history = HistoryTable::default();
        self.main_thread.nodes.reset();
        for t in &mut self.workers {
            t.history = HistoryTable::default();
            t.nodes.reset();
        }
        self.halt.store(false, Ordering::Relaxed);
        self.searching.store(false, Ordering::Relaxed);
    }

    /// This thread creates a number of workers equal to threads - 1. If 4 threads are requested,
    /// the main thread counts as one and then the remaining three are placed in the worker queue.
    pub fn add_workers(
        &mut self,
        threads: usize,
        hash_history: &[u64],
        consts: &'a LmrTable,
        global_nodes: &'a AtomicU64,
    ) {
        self.workers.clear();
        for i in 1..threads {
            self.workers.push(ThreadData::new(
                self.halt,
                hash_history.to_owned(),
                i,
                consts,
                global_nodes,
            ));
        }
    }

    pub fn handle_go(
        &mut self,
        buffer: &[&str],
        board: &Board,
        halt: &AtomicBool,
        msg: &mut Option<String>,
        hash_history: &[u64],
        tt: &TranspositionTable,
    ) {
        self.halt.store(false, Ordering::Relaxed);
        self.main_thread.hash_history = hash_history.to_vec();
        for t in &mut self.workers {
            hash_history.clone_into(&mut t.hash_history);
        }

        if buffer.contains(&"depth") {
            let mut iter = buffer.iter().skip(2);
            let depth = iter.next().unwrap().parse::<i32>().unwrap();
            self.main_thread.search_type = SearchType::Depth(depth);
            for t in &mut self.workers {
                t.search_type = SearchType::Depth(depth);
            }
        } else if buffer.contains(&"nodes") {
            let mut iter = buffer.iter().skip(2);
            let nodes = iter.next().unwrap().parse::<u64>().unwrap();
            self.main_thread.search_type = SearchType::Nodes(nodes);
            for t in &mut self.workers {
                t.search_type = SearchType::Nodes(nodes);
            }
        } else if buffer.contains(&"wtime") {
            let mut clock = parse_time(buffer);
            clock.recommended_time(board.to_move);

            self.main_thread.search_type = SearchType::Time(clock);
        } else if buffer.contains(&"mate") {
            let mut iter = buffer.iter().skip(2);
            let ply = iter.next().unwrap().parse::<i32>().unwrap();
            self.main_thread.search_type = SearchType::Mate(ply);
            for t in &mut self.workers {
                t.search_type = SearchType::Mate(ply);
            }
        } else {
            self.main_thread.search_type = SearchType::Infinite;
            for t in &mut self.workers {
                t.search_type = SearchType::Infinite;
            }
        }

        thread::scope(|s| {
            for t in &mut self.workers {
                s.spawn(|| {
                    search(t, false, *board, tt);
                });
            }

            s.spawn(|| {});
            search(&mut self.main_thread, true, *board, tt);
            self.halt.store(true, Ordering::Relaxed);
            println!("bestmove {}", self.main_thread.best_move.to_san());

            let mut s = String::new();
            let len_read = io::stdin().read_line(&mut s).unwrap();
            if len_read == 0 {
                // Stdin closed, exit for openbench
                exit(0);
            }
            match s.as_str().trim() {
                "isready" => println!("readyok"),
                "quit" => std::process::exit(0),
                "stop" => halt.store(true, Ordering::Relaxed),
                _ => {
                    *msg = Some(s);
                }
            }
        });
        tt.age_up();
    }
}

#[derive(Clone)]
pub struct AtomicCounter<'a> {
    global_nodes: &'a AtomicU64,
    local_nodes: u64,
    batch: u64,
}

const UPDATE_FREQ: u64 = 1024;

impl<'a> AtomicCounter<'a> {
    const fn new(global_nodes: &'a AtomicU64) -> Self {
        Self { global_nodes, local_nodes: 0, batch: 0 }
    }

    pub(crate) fn global_count(&self) -> u64 {
        self.global_nodes.load(Ordering::Relaxed) + self.batch
    }

    pub(crate) const fn local_count(&self) -> u64 {
        self.local_nodes + self.batch
    }

    pub(crate) fn increment(&mut self) {
        self.batch += 1;
        if self.batch > UPDATE_FREQ {
            self.local_nodes += self.batch;
            self.global_nodes.fetch_add(self.batch, Ordering::Relaxed);
            self.batch = 0;
        }
    }

    pub(crate) fn reset(&mut self) {
        self.batch = 0;
        self.local_nodes = 0;
        self.global_nodes.store(0, Ordering::Relaxed);
    }

    pub(crate) const fn check_time(&self) -> bool {
        self.batch == 0
    }
}

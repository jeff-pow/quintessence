use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::{board::Board, pieces::Color, pieces::PieceName, pieces::Piece};

pub struct Move {
    pub starting_idx: i8,
    pub end_idx: i8,
    pub castle: Castle,
}

impl Move {
    pub fn print(&self) {
        print!("Start: {}  End: {}  Castle: ", self.starting_idx, self.end_idx);
        match self.castle {
            Castle::None => { print!("No castle") }
            Castle::WhiteKingCastle => { print!("White King Castle") }
            Castle::WhiteQueenCastle => { print!("White Queen Castle") }
            Castle::BlackKingCastle => { print!("Black King Castle") }
            Castle::BlackQueenCastle => { print!("Black Queen Castle") }
        }
        println!();
    }
}

pub enum Castle {
    None,
    WhiteKingCastle,
    WhiteQueenCastle,
    BlackKingCastle,
    BlackQueenCastle,
}

// Cardinal directions from the point of view of white side
#[derive(EnumIter, Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i8)]
enum Direction {
    North = 8,
    NorthWest = 7,
    West = -1,
    SouthWest = -9,
    South = -8,
    SouthEast = -7,
    East = 1,
    NorthEast = 9,
}

// Takes a direction and number of times that direction is being moved and converts to an x-y tuple
fn convert_idx_to_tuple(d: Direction, repetitions: i8) -> (i8, i8) {
    match d {
        Direction::North => { (0, repetitions)}
        Direction::NorthWest => { (-repetitions, repetitions) }
        Direction::West => { (-repetitions, 0) }
        Direction::SouthWest => { (-repetitions, -repetitions) }
        Direction::South => { (0, -repetitions) }
        Direction::SouthEast => { (repetitions, -repetitions) }
        Direction::East => { (repetitions, 0) }
        Direction::NorthEast => { (repetitions, repetitions) }
    }
}

// Method ensures that two indexes can be added together. Bool determines if operation was
// successful, and i8 contains the added index if boolean is true and nonsense value if false.
// x_sum denotes letters and y_sum denotes number of a square on the board
fn check_index_addition(a: i8, b: (i8, i8)) -> (usize, bool) {
    let a_x_component = a / 8;
    let a_y_component = a % 8;
    let b_x_component = b.0;
    let b_y_component = b.1;
    let x_sum = a_x_component + b_x_component;
    let y_sum = a_y_component + b_y_component;
    if !(0..8).contains(&x_sum) || !(0..8).contains(&y_sum) {
        return (0, false);
    }
    let ret = (a + b.0 * 8 + b.1) as usize;
    (ret, true)
    /*
    match piece.piece_name {
        PieceName::King => {
            if !(0..8).contains(&x_sum) || !(0..8).contains(&y_sum) {
                return (0, false);
            }
            let ret = (a + b.0 * 8 + b.1) as usize;
            (ret, true)
        }
        PieceName::Queen => {
            if !(0..8).contains(&x_sum) || !(0..8).contains(&y_sum) {
                return (0, false);
            }
            let ret = (a + b.0 * 8 + b.1) as usize;
            (ret, true)
        }
        PieceName::Rook => {
            if !(0..8).contains(&x_sum) || !(0..8).contains(&y_sum) {
                return (0, false);
            }
            let ret = (a + b.0 * 8 + b.1) as usize;
            (ret, true)
        }
        PieceName::Bishop => {
            if !(0..8).contains(&x_sum) || !(0..8).contains(&y_sum) {
                return (0, false);
            }
            let ret = (a + b.0 * 8 + b.1) as usize;
            (ret, true)
        }
        PieceName::Knight => {
            if !(0..8).contains(&x_sum) || !(0..8).contains(&y_sum) {
                return (0, false);
            }
            let ret = (a + b.0 * 8 + b.1) as usize;
            (ret, true)
        }
        PieceName::Pawn => {
            if !(0..8).contains(&x_sum) || !(0..8).contains(&y_sum) {
                return (0, false);
            }
            let ret = (a + b.0 * 8 + b.1) as usize;
            (ret, true)
        }
    }
     */
}

// Method returns a tuple with a bool stating if a piece is on that square and the color of the
// piece if there is a piece
fn check_space_occupancy(board: &Board, piece: &Piece, potential_space: i8) -> (bool, Color) {
    match board.board[potential_space as usize] {
        None => return (false, Color::White),
        Some(_piece) => {
            let _p = board.board[potential_space as usize].unwrap();
        }
    }
    if board.board[potential_space as usize].is_none() {
        return (false, Color::White);
    }
    (true, board.board[potential_space as usize].unwrap().color)
}

pub fn generate_all_moves(board: &Board) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::new();
    let color_to_move = board.to_move;
    for piece in board.board {
        match piece {
            None => continue,
            Some(piece) => {
                if piece.color == color_to_move {
                    let mut vec = generate_moves_for_piece(board, &piece);
                    moves.append(&mut vec);
                }
            }
        }
    }
    moves
}

fn generate_moves_for_piece(board: &Board, piece: &Piece) -> Vec<Move> {
    match piece.piece_name {
        PieceName::King => generate_king_moves(board, piece),
        PieceName::Queen => generate_queen_moves(board, piece),
        PieceName::Rook => generate_rook_moves(board, piece),
        PieceName::Bishop => generate_bishop_moves(board, piece),
        PieceName::Knight => generate_knight_moves(board, piece),
        PieceName::Pawn => generate_pawn_moves(board, piece),
    }
}

fn generate_king_moves(board: &Board, piece: &Piece) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::new();
    // Generate moves for castling
    match piece.color {
        Color::White => {
            if board.white_queen_castle && piece.current_square == 4 && board.board[3].is_none() && board.board[2].is_none() &&
                board.board[1].is_none() {
                    moves.push(Move {
                        starting_idx: 4,
                        end_idx: 2,
                        castle: Castle::WhiteQueenCastle,
                    });
            }
            if board.white_king_castle && piece.current_square == 4 && board.board[5].is_none() && board.board[6].is_none() {
                moves.push(Move {
                    starting_idx: 4,
                    end_idx: 6,
                    castle: Castle::WhiteKingCastle,
                });
            }
        }
        Color::Black => {
            if board.black_queen_castle && piece.current_square == 60 && board.board[57].is_none() && board.board[58].is_none() &&
                board.board[59].is_none() {
                    moves.push(Move {
                        starting_idx: 60,
                        end_idx: 58,
                        castle: Castle::BlackQueenCastle,
                    });
            }
            if board.black_king_castle && piece.current_square == 60 && board.board[61].is_none() && board.board[62].is_none() {
                moves.push(Move {
                    starting_idx: 60, 
                    end_idx: 62,
                    castle: Castle::BlackKingCastle,
                });
            }
        }
    }
    let current_idx = piece.current_square;
    for direction in Direction::iter() {
        // Function contains a bool determining if direction points to a valid square based off the
        // current square, and if true the usize is the new index being looked at
        let new_square_indices = convert_idx_to_tuple(direction, 1);
        let (idx, square_validity) = check_index_addition(piece.current_square, new_square_indices);
        if !square_validity {
            continue;
        }
        // Method returns a tuple containing a bool determining if potential square contains a
        // piece of any kind, and if true color contains the color of the new piece
        let occupancy = check_space_occupancy(board, piece, idx as i8);
        if !occupancy.0 {
            // If position not occupied, add the move
            moves.push(Move {
                starting_idx: current_idx,
                end_idx: idx as i8,
                castle: Castle::None,
            });
        }
        // Otherwise square is occupied
        else {
            if piece.color == occupancy.1 {
                // If color of other piece is the same as current piece, you can't move there
                continue;
            }
            // Otherwise you can capture that piece
            moves.push(Move {
                starting_idx: current_idx,
                end_idx: idx as i8,
                castle: Castle::None,
            });
        }
    }
    moves
}

fn generate_queen_moves(board: &Board, piece: &Piece) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::new();
    for direction in Direction::iter() {
        for i in 1..8 {
            let new_square_indices = convert_idx_to_tuple(direction, i as i8);
            let (idx, square_validity) = check_index_addition(piece.current_square, new_square_indices);
            if !square_validity {
                break;
            }
            let (occupied, potential_color) = check_space_occupancy(board, piece, idx as i8);
            if !occupied {
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: idx as i8,
                    castle: Castle::None,
                });
            }
            else {
                if potential_color == piece.color {
                    break;
                }
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: idx as i8,
                    castle: Castle::None,
                });
                break;
            }
        }
    }
    moves
}

fn generate_rook_moves(board: &Board, piece: &Piece) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::new();
    return moves;
    for direction in Direction::iter() {
        match direction {
            // Filter out the four main cardinal directions
            Direction::NorthWest => continue,
            Direction::NorthEast => continue,
            Direction::SouthWest => continue,
            Direction::SouthEast => continue,
            // Continue generating move if move is diagonal
            _ => (),
        }
        for i in 1..8 {
            let new_square_indices = convert_idx_to_tuple(direction, i as i8);
            let (idx, square_validity) = check_index_addition(piece.current_square, new_square_indices);
            if !square_validity {
                break;
            }
            let (occupied, potential_color) = check_space_occupancy(board, piece, idx as i8);
            if !occupied {
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: idx as i8,
                    castle: Castle::None,
                });
            }
            else {
                if potential_color == piece.color {
                    break;
                }
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: idx as i8,
                    castle: Castle::None,
                });
                break;
            }
        }
    }
    moves
}

fn generate_bishop_moves(board: &Board, piece: &Piece) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::new();
    for direction in Direction::iter() {
        match direction {
            // Filter out the four main cardinal directions
            Direction::North => continue,
            Direction::South => continue,
            Direction::East => continue,
            Direction::West => continue,
            // Continue generating move if move is diagonal
            _ => (),
        }
        for i in 1..8 {
            let new_square_indices = convert_idx_to_tuple(direction, i as i8);
            let (idx, square_validity) = check_index_addition(piece.current_square, new_square_indices);
            if !square_validity {
                break;
            }
            let (occupied, potential_color) = check_space_occupancy(board, piece, idx as i8);
            if !occupied {
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: idx as i8,
                    castle: Castle::None,
                });
            }
            else {
                if potential_color == piece.color {
                    break;
                }
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: idx as i8,
                    castle: Castle::None,
                });
                break;
            }
        }
    }
    moves
}

// Movement chords are defined by a combination of three cardinal directions - ex West West North
#[derive(EnumIter, Copy, Clone, Debug, PartialEq, Eq)]
enum KnightMovement {
    WWN = 6,
    WNN = 15,
    ENN = 17,
    EEN = 10,
    EES = -6,
    ESS = -15,
    WSS = -17,
    WWS = -10,
}

fn knight_move_to_tuple(k: KnightMovement) -> (i8, i8) {
    match k {
        KnightMovement::WWN => { (-2, 1) }
        KnightMovement::WNN => { (-1, 2) }
        KnightMovement::ENN => { (1, 2) }
        KnightMovement::EEN => { (2, 1) }
        KnightMovement::EES => { (2, -1) }
        KnightMovement::ESS => { (1, -2) }
        KnightMovement::WSS => { (-1, -2) }
        KnightMovement::WWS => { (-2, -1) }
    }
}

fn generate_knight_moves(board: &Board, piece: &Piece) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::new();
    for direction in KnightMovement::iter() {
        let movement_directions = knight_move_to_tuple(direction);
        let square_validity = check_index_addition(piece.current_square, movement_directions);
        if !square_validity.1 {
            continue;
        }
        let (occupied, potential_color) = check_space_occupancy(board, piece, square_validity.0 as i8);
        if !occupied {
            moves.push(Move {
                starting_idx: piece.current_square,
                end_idx: square_validity.0 as i8,
                castle: Castle::None,
            });
        }
        else {
            if piece.color == potential_color {
                continue;
            }
            moves.push(Move {
                starting_idx: piece.current_square,
                end_idx: square_validity.0 as i8,
                castle: Castle::None,
            });
        }
    }   
    moves
}

fn generate_pawn_moves(board: &Board, piece: &Piece) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::new();
    match piece.color {
        Color::White => {
            // Determines if one square in front of piece is occupied
            let (n_occupied, _) = check_space_occupancy(board, piece, piece.current_square + Direction::North as i8);
            // Determines if two squares in front of piece is occupied
            let (nn_occupied, _) = check_space_occupancy(board, piece, piece.current_square + 2 * Direction::North as i8);
            if piece.current_square > 7 && piece.current_square < 15 && !n_occupied && !nn_occupied {
                // Handles moving two spaces forward if pawn has not moved yet
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: piece.current_square + 2 * Direction::North as i8,
                    castle: Castle::None,
                });
            }
            if !n_occupied {
                // Can still move one space forward on the first square
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: piece.current_square + Direction::North as i8,
                    castle: Castle::None,
                });
            }
            let (nw_occupied, potential_color) = check_space_occupancy(board, piece, piece.current_square + Direction::NorthWest as i8);
            if nw_occupied && piece.color != potential_color {
                // Capturing to the northwest
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: piece.current_square + Direction::NorthWest as i8,
                    castle: Castle::None,
                });
            }
            let (ne_occupied, potential_color) = check_space_occupancy(board, piece, piece.current_square + Direction::NorthEast as i8);
            if ne_occupied && piece.color != potential_color {
                // Capturing to the northeast
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: piece.current_square + Direction::NorthEast as i8,
                    castle: Castle::None,
                });
            }
        }
        Color::Black => {
            // First square to the south
            let (s_occupied, _) = check_space_occupancy(board, piece, piece.current_square + Direction::South as i8);
            // Second square to the south
            let (ss_occupied, _) = check_space_occupancy(board, piece, piece.current_square + 2 * Direction::South as i8);
            if piece.current_square > 47 && piece.current_square < 56 && !s_occupied && !ss_occupied {
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: piece.current_square + 2 * Direction::South as i8,
                    castle: Castle::None,
                });
            }
            if !s_occupied {
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: piece.current_square + Direction::South as i8,
                    castle: Castle::None,
                });
            }
            let (se_occupied, potential_color) = check_space_occupancy(board, piece, piece.current_square + Direction::SouthEast as i8);
            if se_occupied && piece.color != potential_color {
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: piece.current_square + Direction::SouthEast as i8,
                    castle: Castle::None,
                });
            }
            let (sw_occupied, potential_color) = check_space_occupancy(board, piece, piece.current_square + Direction::SouthWest as i8);
            if sw_occupied && piece.color != potential_color {
                moves.push(Move {
                    starting_idx: piece.current_square,
                    end_idx: piece.current_square + Direction::SouthWest as i8,
                    castle: Castle::None,
                });
            }
        }
    }
    moves
}


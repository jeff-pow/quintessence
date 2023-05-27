# chess-engine

A reasonably capable chess engine that was written as a personal project and learning experience. 

### Features
- Utilizes the UCI framework
- Alpha-beta pruning to make searching the game tree more efficient
- Zobrist hashing to store the scoring of previously found board states in a transposition table

Can be built by entering the chess-engine directory and running `cargo b --release`. A binary will be generated that can be used in 
any UCI compliant chess GUI.

### Useful Links
- https://www.chessprogramming.org/Main_Page
- https://lichess.org/editor

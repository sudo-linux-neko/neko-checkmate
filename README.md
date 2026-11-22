# Neko Meow Chess Engine v1.0

A CPU-only C++ chess engine with NNUE-style neural network evaluation and self-play training.

## Features

- **Bitboard representation** with ray-based sliding piece attacks
- **Alpha-beta search** with iterative deepening, transposition tables, move ordering, quiescence search
- **Small NNUE-style neural network** (768→128→64→1) for position evaluation
- **Self-play training** that generates training data and improves the network
- **Checkpoint/resume** - safely stop and resume training without losing progress
- **UCI protocol** support for use with chess GUIs
- **CPU-only** - optimized for Intel Core i5-1335U (10 cores/12 threads)

## Quick Start

### 1. Build the Engine

```bash
cd /home/neko/neko-checkmate
make
```

This compiles with `-O3 -march=native` optimizations.

### 2. Run Benchmark

```bash
./neko_meow bench
```

### 3. Start Self-Play Training

```bash
./neko_meow selfplay --depth 6 --games 20 --iterations 100
```

### 4. Stop Training Safely

Press `Ctrl+C` at any time. The engine will save a checkpoint.

### 5. Resume Training

```bash
./neko_meow selfplay
```

Training resumes from the last checkpoint automatically.

## Command Line Options

```
neko_meow uci              # Start UCI protocol mode
neko_meow selfplay         # Start self-play training
neko_meow train            # Train NNUE on existing data
neko_meow bench            # Run benchmark
```

### Self-Play Options

| Option | Default | Description |
|--------|---------|-------------|
| `--games N` | 20 | Games per iteration |
| `--depth N` | 6 | Search depth |
| `--iterations N` | 100 | Total training iterations |
| `--save-every N` | 5 | Save checkpoint every N iterations |
| `--time N` | 500 | Time per move (ms) |

## Project Structure

```
neko-checkmate/
├── Makefile              # Build system
├── CMakeLists.txt        # Alternative CMake build
├── README.md             # This file
├── src/
│   ├── types.h           # Basic types and constants
│   ├── bitboard.h/cpp    # Bitboard operations
│   ├── board.h/cpp       # Board representation
│   ├── movegen.h/cpp     # Move generation
│   ├── nnue.h/cpp        # Neural network implementation
│   ├── search.h/cpp      # Alpha-beta search
│   ├── evaluate.h/cpp    # Evaluation
│   ├── selfplay.h/cpp    # Self-play loop
│   ├── training.h/cpp    # Training logic
│   ├── uci.h/cpp         # UCI protocol
│   └── main.cpp          # Entry point
└── checkpoint/           # Training checkpoints (auto-created)
    ├── nnue.bin          # Network weights
    ├── optimizer.bin     # Adam optimizer state
    ├── selfplay_state.bin # Training progress
    └── training_data.txt # Collected positions
```

## How Training Works

1. **Self-Play**: The engine plays games against itself using the current network
2. **Data Collection**: Positions are saved with engine evaluations and game results
3. **Training**: The NNUE network is trained on collected positions using backpropagation
4. **Checkpoint**: Network weights, optimizer state, and progress are saved
5. **Resume**: On next run, everything loads automatically from checkpoint

## Using with UCI GUI

```bash
./neko_meow uci
```

Then connect your chess GUI (Arena, BanksiaGUI, etc.) to the engine.

## Hardware Optimization

- **CPU**: Optimized for hybrid P-core/E-core architecture
- **Memory**: ~100MB for transposition table + network weights
- **Threading**: Single-threaded search (can be extended)
- **SIMD**: SSE4.1 and POPCNT instructions enabled

## Further Resources

- [Stockfish NNUE](https://www.chessprogramming.org/NNUE) - Chess programming wiki on NNUE
- [Jet Chess Engine](https://github.com/rafid-dev/jet) - Lightweight NNUE engine reference
- [Lc0](https://github.com/LeelaChessZero/lc0) - Neural network chess engine
- [Chess Programming Wiki](https://www.chessprogramming.org/) - Comprehensive resource

## License

This project is provided as-is for educational purposes.

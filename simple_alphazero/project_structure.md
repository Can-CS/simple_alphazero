# Simple AlphaZero Chess Bot - Project Structure

## Overview

This project implements a simplified AlphaZero-style chess bot that combines Monte Carlo Tree Search (MCTS) with a lightweight neural network. The code is designed to be clean, maintainable, and optimized for reaching 800-1400 ELO in 10-20 hours of training on an RTX 4070.

## Project Structure

```
simple_alphazero/
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── main.py                   # Entry point for training and playing
├── config.py                 # Configuration parameters
├── model/
│   ├── __init__.py
│   ├── neural_net.py         # Lightweight neural network
│   └── checkpoint.py         # Model saving and loading
├── mcts/
│   ├── __init__.py
│   ├── node.py               # MCTS node implementation
│   ├── search.py             # MCTS search algorithm
│   └── evaluation.py         # Position evaluation (heuristic + NN)
├── chess_env/
│   ├── __init__.py
│   ├── board_representation.py  # Chess board representation
│   └── move_encoding.py      # Move encoding/decoding
├── training/
│   ├── __init__.py
│   ├── self_play.py          # Self-play game generation
│   ├── optimization.py       # Neural network training
│   └── replay_buffer.py      # Experience replay buffer
├── gui/
│   ├── __init__.py
│   ├── chess_board.py        # Chess board visualization
│   ├── game_viewer.py        # Game replay functionality
│   └── training_dashboard.py # Training progress visualization
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Logging functionality
│   └── profiler.py           # Performance profiling
└── tests/                    # Basic tests for core functionality
    ├── __init__.py
    ├── test_mcts.py
    └── test_neural_net.py
```

## Key Components

### 1. Neural Network (model/neural_net.py)
A lightweight convolutional neural network with:
- 16-channel input (standard chess representation)
- 2-3 convolutional layers with batch normalization
- Dual policy and value heads
- Designed for fast inference on RTX 4070

### 2. MCTS Implementation (mcts/)
A clean, efficient MCTS implementation with:
- UCB1 formula for node selection
- Dirichlet noise at root for exploration
- Parallel simulations for performance
- Integration with neural network for position evaluation

### 3. Training Pipeline (training/)
A streamlined training process with:
- Self-play game generation
- Experience replay buffer
- Neural network optimization
- Regular checkpointing

### 4. GUI and Visualization (gui/)
A simple but effective visualization system with:
- Interactive chess board
- Game replay functionality
- Training progress visualization

### 5. Logging and Monitoring (utils/)
Comprehensive logging with:
- Clear, structured log format
- Performance metrics tracking
- Error detection and reporting

## Design Principles

1. **Simplicity**: Minimal, readable code with clear documentation
2. **Maintainability**: Modular design with well-defined interfaces
3. **Performance**: Optimized for RTX 4070 to reach target ELO in specified time
4. **Robustness**: Error handling and recovery mechanisms
5. **Observability**: Comprehensive logging and monitoring

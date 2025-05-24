# Simple AlphaZero Chess Bot

A simplified AlphaZero-style chess bot implementation that combines Monte Carlo Tree Search (MCTS) with a lightweight neural network. This project is designed to be clean, maintainable, and optimized for reaching 800-1400 ELO in 10-20 hours of training on an RTX 4070.

## Features

- **Lightweight Neural Network**: A simplified convolutional neural network with dual policy and value heads
- **Efficient MCTS Implementation**: Monte Carlo Tree Search with UCB1 formula and Dirichlet noise for exploration
- **Self-Play Training**: Generate training data through self-play games
- **Interactive GUI**: Play against the trained model or watch self-play games
- **Comprehensive Logging**: Clear, structured logs and performance metrics tracking
- **RTX 4070 Optimization**: Configured for optimal performance on RTX 4070 GPU
- **Checkpointing**: Regular model saving and loading for training continuity

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Chess 1.9+
- Additional dependencies in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

To train a new model:

```
python main.py train
```

To continue training from a checkpoint:

```
python main.py train --load-model checkpoints/model_iter_50.pt
```

### Playing

To play against the trained model:

```
python main.py play --model checkpoints/best_model.pt --play-as white
```

## Project Structure

```
simple_alphazero/
├── model/
│   ├── neural_net.py         # Lightweight neural network
├── mcts/
│   ├── node.py               # MCTS node implementation
│   ├── search.py             # MCTS search algorithm
├── training/
│   ├── self_play.py          # Self-play game generation
│   ├── optimization.py       # Neural network training
├── gui/
│   ├── chess_board.py        # Chess board visualization
├── utils/
│   ├── logger.py             # Logging functionality
├── config.py                 # Configuration parameters
├── main.py                   # Entry point for training and playing
├── requirements.txt          # Dependencies
```

## Configuration

All configurable parameters are in `config.py`. Key parameters include:

- Neural network architecture
- MCTS simulation count
- Training iterations and batch size
- Learning rate and regularization
- GUI settings

## Performance

The model is designed to reach 800-1400 ELO in 10-20 hours of training on an RTX 4070. Performance metrics are logged during training and can be visualized using the built-in plotting functionality.

## Acknowledgements

This project is inspired by the AlphaZero algorithm by DeepMind, but significantly simplified for faster training and better maintainability.

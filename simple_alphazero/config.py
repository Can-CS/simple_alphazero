"""
Configuration parameters for the chess bot.

This module contains all configurable parameters for the chess bot,
making it easy to adjust settings without modifying code.
"""

# Neural Network Configuration
NETWORK_CONFIG = {
    'num_channels': 64,       # Number of channels in convolutional layers
    'batch_norm': True,       # Whether to use batch normalization
    'device': 'cuda',         # Device to use ('cuda' or 'cpu')
}

# MCTS Configuration
MCTS_CONFIG = {
    'num_simulations': 800,   # Number of MCTS simulations per move
    'c_puct': 1.5,            # Exploration constant in PUCT formula
    'dirichlet_alpha': 0.3,   # Alpha parameter for Dirichlet noise
    'dirichlet_epsilon': 0.25,# Weight of Dirichlet noise at root
}

# Training Configuration
TRAINING_CONFIG = {
    'num_iterations': 100,    # Number of training iterations
    'num_self_play_games': 50,# Number of self-play games per iteration
    'num_epochs': 10,         # Number of training epochs per iteration
    'batch_size': 256,        # Batch size for training
    'learning_rate': 0.001,   # Learning rate for optimizer
    'weight_decay': 1e-4,     # L2 regularization
    'checkpoint_interval': 5, # Save checkpoint every N iterations
    'max_moves': 512,         # Maximum moves per game
    'temperature_schedule': { # Temperature schedule for move selection
        0: 1.0,               # Start with high temperature
        30: 0.5,              # Reduce after opening
        60: 0.25,             # Further reduce in middlegame
    },
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'num_evaluation_games': 10,# Number of games to play for evaluation
    'opponent_levels': [0, 1, 2, 3],  # Stockfish levels to evaluate against
    'time_per_move': 0.1,     # Time per move for Stockfish (seconds)
}

# GUI Configuration
GUI_CONFIG = {
    'width': 600,             # Window width
    'height': 650,            # Window height (extra space for status)
    'fps': 30,                # Frames per second
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_dir': 'logs',        # Directory for log files
    'plot_dir': 'plots',      # Directory for plots
    'log_level': 'INFO',      # Logging level
}

# File Paths
PATHS = {
    'checkpoint_dir': 'checkpoints',  # Directory for model checkpoints
    'best_model': 'checkpoints/best_model.pt',  # Path to best model
    'latest_model': 'checkpoints/latest_model.pt',  # Path to latest model
}

# RTX 4070 Optimization
RTX4070_OPTIMIZATIONS = {
    'mixed_precision': True,    # Whether to use mixed precision training
    'num_workers': 4,           # Number of dataloader workers
    'pin_memory': True,         # Whether to pin memory in dataloader
    'cudnn_benchmark': True,    # Whether to use cudnn benchmark
    'use_torch_compile': False,  # Disabled due to Triton dependency issues
    'compile_mode': 'reduce-overhead',  # Compile mode: 'default', 'reduce-overhead', or 'max-autotune'
    'compile_backend': 'inductor',  # Backend to use: 'inductor' (doesn't require Triton)
    'use_cuda_graphs': True,    # Whether to use CUDA graphs for repeated operations
    'cuda_graph_batch_size': 256,  # Batch size to use for CUDA graph capture
    'warmup_iterations': 3,     # Number of warmup iterations before capturing CUDA graph
    'capture_error_mode': 'thread_local',  # CUDA graph capture error mode: 'global' or 'thread_local'
    'enable_cuda_dsa': False,   # Enable CUDA device-side assertions (helps with debugging)
    'skip_first_backward_pass': True,  # Skip first backward pass in graph capture to avoid issues
}


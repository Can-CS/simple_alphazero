"""
Main entry point for the chess bot.

This module provides the main functionality for training, playing, and evaluating
the AlphaZero-style chess bot.
"""

import os
import time
import torch
import chess
import argparse
import numpy as np
from torch.optim import Adam

# Import project modules
from model.neural_net import ChessNet, create_model
from mcts.search import MCTS
from training.self_play import SelfPlay
from training.optimization import Trainer
from gui.chess_board import ChessGUI, AIPlayer
from utils.logger import Logger, Profiler
import config

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

def train(args):
    """
    Train the chess bot using self-play and neural network learning.

    Args:
        args: Command-line arguments
    """
    # Set up logger
    logger = Logger(log_dir=config.LOGGING_CONFIG['log_dir'])
    logger.log_info("Starting training process")

    # Set up profiler
    profiler = Profiler()

    # Create directories
    os.makedirs(config.PATHS['checkpoint_dir'], exist_ok=True)

    # Set device
    device = torch.device(config.NETWORK_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.log_info(f"Using device: {device}")

    # RTX 4070 optimizations
    if device.type == 'cuda':
        # Enable cuDNN benchmark
        if config.RTX4070_OPTIMIZATIONS['cudnn_benchmark']:
            torch.backends.cudnn.benchmark = True
            logger.log_info("CUDNN benchmark enabled for RTX 4070 optimization")

        # Log torch.compile status
        if config.RTX4070_OPTIMIZATIONS['use_torch_compile']:
            logger.log_info(f"torch.compile enabled with mode: {config.RTX4070_OPTIMIZATIONS.get('compile_mode', 'default')}")

        # Log CUDA graph status
        if config.RTX4070_OPTIMIZATIONS['use_cuda_graphs']:
            logger.log_info("CUDA graphs enabled for repeated operations")
            logger.log_info(f"CUDA graph batch size: {config.RTX4070_OPTIMIZATIONS.get('cuda_graph_batch_size', 256)}")
            logger.log_info(f"CUDA graph warmup iterations: {config.RTX4070_OPTIMIZATIONS.get('warmup_iterations', 3)}")

    # Create or load model
    if args.load_model and os.path.exists(args.load_model):
        # Load existing model
        logger.log_info(f"Loading model from {args.load_model}")
        model = create_model(device=device)
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_iteration = checkpoint.get('iteration', 0) + 1
    else:
        # Create new model
        logger.log_info("Creating new model")
        model = create_model(device=device)
        start_iteration = 1

    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.TRAINING_CONFIG['learning_rate'],
        weight_decay=config.TRAINING_CONFIG['weight_decay']
    )

    # Create trainer with mixed precision if on CUDA
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        num_epochs=config.TRAINING_CONFIG['num_epochs'],
        checkpoint_dir=config.PATHS['checkpoint_dir'],
        use_mixed_precision=config.RTX4070_OPTIMIZATIONS['mixed_precision']
    )

        # Dynamic simulation schedule
    def get_dynamic_simulations(iteration):
        base = config.MCTS_CONFIG['num_simulations']
        return max(100, int(base * min(1.0, iteration / 500)))  # Gradually ramp up to base

    # Training loop
    for iteration in range(start_iteration, start_iteration + config.TRAINING_CONFIG['num_iterations']):
        logger.log_info(f"Starting iteration {iteration}")

        # Update simulations dynamically
        num_simulations = get_dynamic_simulations(iteration)
        logger.log_info(f"Using {num_simulations} MCTS simulations")

        mcts = MCTS(
            model=model,
            device=device,
            num_simulations=num_simulations,
            dirichlet_alpha=config.MCTS_CONFIG['dirichlet_alpha'],
            dirichlet_epsilon=config.MCTS_CONFIG['dirichlet_epsilon'],
            c_puct=config.MCTS_CONFIG['c_puct']
        )

        self_play = SelfPlay(
            model=model,
            device=device,
            num_simulations=num_simulations,
            max_moves=config.TRAINING_CONFIG['max_moves'],
            temperature_schedule=config.TRAINING_CONFIG['temperature_schedule']
        )


    # Training loop
    for iteration in range(start_iteration, start_iteration + config.TRAINING_CONFIG['num_iterations']):
        logger.log_info(f"Starting iteration {iteration}")

        # Generate self-play games
        examples = []
        profiler.start('self_play')

        for game_idx in range(config.TRAINING_CONFIG['num_self_play_games']):
            # Play a game
            game_examples, result = self_play.play_game()
            examples.extend(game_examples)

            # Log game result
            game_length = len(game_examples) // 2  # Approximate number of moves
            logger.log_game(game_idx, result, game_length)

        self_play_time = profiler.stop('self_play')
        logger.log_info(f"Self-play completed: {len(examples)} examples generated in {self_play_time:.2f}s")

        # Train the network
        profiler.start('training')
        loss = trainer.train(examples)
        training_time = profiler.stop('training')

        # Log training results
        logger.log_training(
            iteration=iteration,
            policy_loss=trainer.policy_losses[-1],
            value_loss=trainer.value_losses[-1],
            total_loss=loss,
            time_taken=training_time
        )

        # Save checkpoint
        if iteration % config.TRAINING_CONFIG['checkpoint_interval'] == 0:
            checkpoint_path = trainer.save_checkpoint(iteration)
            logger.log_info(f"Checkpoint saved to {checkpoint_path}")

            # Also save as latest model
            trainer.save_checkpoint(iteration, filename="latest_model.pt")

        # Estimate ELO (simplified)
        estimated_elo = 800 + (iteration * 5)  # Very rough estimate, replace with actual evaluation
        logger.log_elo_estimate(iteration, estimated_elo)

        # Save metrics and plots
        logger.save_metrics()
        logger.plot_training_progress(save_dir=config.LOGGING_CONFIG['plot_dir'])

        # Log profiler summary
        summary = profiler.summary()
        for name, stats in summary.items():
            logger.log_info(f"Profiler - {name}: "
                           f"Avg: {stats['average']:.2f}s, "
                           f"Total: {stats['total']:.2f}s, "
                           f"Count: {stats['count']}")

    # Final checkpoint
    final_path = trainer.save_checkpoint(
        start_iteration + config.TRAINING_CONFIG['num_iterations'] - 1,
        filename="final_model.pt"
    )
    logger.log_info(f"Training completed. Final model saved to {final_path}")


def play(args):
    """
    Play against the trained chess bot.

    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device(config.NETWORK_CONFIG['device'] if torch.cuda.is_available() else 'cpu')

    # RTX 4070 optimizations for inference
    if device.type == 'cuda':
        # Enable cuDNN benchmark
        if config.RTX4070_OPTIMIZATIONS['cudnn_benchmark']:
            torch.backends.cudnn.benchmark = True
            print("CUDNN benchmark enabled for RTX 4070 optimization")

    # Load model
    model_path = args.model if args.model else config.PATHS['best_model']
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    model = create_model(device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Print model optimization status
    if device.type == 'cuda':
        if config.RTX4070_OPTIMIZATIONS['use_torch_compile']:
            print(f"Using torch.compile with mode: {config.RTX4070_OPTIMIZATIONS.get('compile_mode', 'default')}")
        if config.RTX4070_OPTIMIZATIONS['use_cuda_graphs']:
            print("CUDA graphs enabled for inference")

    # Create MCTS
    mcts = MCTS(
        model=model,
        device=device,
        num_simulations=args.simulations if args.simulations else config.MCTS_CONFIG['num_simulations'],
        c_puct=config.MCTS_CONFIG['c_puct']
    )

    # Create AI player
    ai_color = chess.BLACK if args.play_as.lower() == 'white' else chess.WHITE
    ai_player = AIPlayer(mcts, color=ai_color)

    # Create GUI
    gui = ChessGUI(
        width=config.GUI_CONFIG['width'],
        height=config.GUI_CONFIG['height']
    )

    # Run GUI
    gui.run(ai_player=ai_player)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Simple AlphaZero Chess Bot')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the chess bot')
    train_parser.add_argument('--load-model', type=str, help='Path to model to continue training')

    # Play command
    play_parser = subparsers.add_parser('play', help='Play against the chess bot')
    play_parser.add_argument('--model', type=str, help='Path to model file')
    play_parser.add_argument('--play-as', type=str, default='white', choices=['white', 'black'],
                            help='Play as white or black')
    play_parser.add_argument('--simulations', type=int, help='Number of MCTS simulations per move')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'play':
        play(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()





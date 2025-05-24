"""
Logging and monitoring utilities.

This module provides logging functionality for training and gameplay,
with clear, structured log formats and performance metrics tracking.
"""

import logging
import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class Logger:
    """
    Logger for chess training and gameplay.
    
    This class handles:
    - Setting up logging to file and console
    - Tracking and saving performance metrics
    - Generating training reports
    """
    
    def __init__(self, log_dir='logs', log_level=logging.INFO):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save log files
            log_level: Logging level
        """
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('chess_bot')
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create timestamp for log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # File handler
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Performance metrics
        self.metrics = {
            'training_iterations': [],
            'game_lengths': [],
            'game_results': [],
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'mcts_times': [],
            'training_times': [],
            'elo_estimates': [],
        }
        
        # Metrics file
        self.metrics_file = os.path.join(log_dir, f'metrics_{timestamp}.json')
        
        self.log_info(f"Logger initialized. Logs will be saved to {log_file}")
    
    def log_info(self, message):
        """Log an info message."""
        self.logger.info(message)
    
    def log_warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_error(self, message):
        """Log an error message."""
        self.logger.error(message)
    
    def log_game(self, iteration, result, length, player_color=None):
        """
        Log a completed game.
        
        Args:
            iteration: Current training iteration
            result: Game result (1.0, 0.0, or -1.0)
            length: Number of moves in the game
            player_color: Color of the player (if playing against AI)
        """
        # Convert result to string
        if result == 1.0:
            result_str = "White win"
        elif result == -1.0:
            result_str = "Black win"
        else:
            result_str = "Draw"
        
        # Log game result
        if player_color is not None:
            player_str = "White" if player_color else "Black"
            self.log_info(f"Game {iteration}: {result_str} in {length} moves (Player: {player_str})")
        else:
            self.log_info(f"Game {iteration}: {result_str} in {length} moves")
        
        # Update metrics
        self.metrics['game_results'].append(result)
        self.metrics['game_lengths'].append(length)
    
    def log_training(self, iteration, policy_loss, value_loss, total_loss, time_taken):
        """
        Log training metrics.
        
        Args:
            iteration: Current training iteration
            policy_loss: Policy head loss
            value_loss: Value head loss
            total_loss: Combined loss
            time_taken: Time taken for training
        """
        self.log_info(f"Training {iteration}: "
                     f"Policy Loss: {policy_loss:.4f}, "
                     f"Value Loss: {value_loss:.4f}, "
                     f"Total Loss: {total_loss:.4f}, "
                     f"Time: {time_taken:.2f}s")
        
        # Update metrics
        self.metrics['training_iterations'].append(iteration)
        self.metrics['policy_losses'].append(policy_loss)
        self.metrics['value_losses'].append(value_loss)
        self.metrics['total_losses'].append(total_loss)
        self.metrics['training_times'].append(time_taken)
    
    def log_mcts_stats(self, num_simulations, time_taken):
        """
        Log MCTS performance statistics.
        
        Args:
            num_simulations: Number of MCTS simulations
            time_taken: Time taken for MCTS search
        """
        sims_per_second = num_simulations / max(time_taken, 0.001)
        self.log_info(f"MCTS: {num_simulations} simulations in {time_taken:.2f}s "
                     f"({sims_per_second:.1f} sims/s)")
        
        # Update metrics
        self.metrics['mcts_times'].append(time_taken)
    
    def log_elo_estimate(self, iteration, elo):
        """
        Log estimated ELO rating.
        
        Args:
            iteration: Current training iteration
            elo: Estimated ELO rating
        """
        self.log_info(f"Iteration {iteration}: Estimated ELO: {elo}")
        
        # Update metrics
        self.metrics['elo_estimates'].append((iteration, elo))
    
    def save_metrics(self):
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f)
        
        self.log_info(f"Metrics saved to {self.metrics_file}")
    
    def plot_training_progress(self, save_dir='plots'):
        """
        Plot training progress and save figures.
        
        Args:
            save_dir: Directory to save plots
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot losses
        if self.metrics['total_losses']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['training_iterations'], self.metrics['policy_losses'], label='Policy Loss')
            plt.plot(self.metrics['training_iterations'], self.metrics['value_losses'], label='Value Loss')
            plt.plot(self.metrics['training_iterations'], self.metrics['total_losses'], label='Total Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True)
            
            # Save figure
            loss_plot_path = os.path.join(save_dir, 'training_losses.png')
            plt.savefig(loss_plot_path)
            plt.close()
            
            self.log_info(f"Training losses plot saved to {loss_plot_path}")
        
        # Plot game lengths
        if self.metrics['game_lengths']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['game_lengths'])
            plt.xlabel('Game')
            plt.ylabel('Number of Moves')
            plt.title('Game Lengths')
            plt.grid(True)
            
            # Save figure
            length_plot_path = os.path.join(save_dir, 'game_lengths.png')
            plt.savefig(length_plot_path)
            plt.close()
            
            self.log_info(f"Game lengths plot saved to {length_plot_path}")
        
        # Plot ELO estimates
        if self.metrics['elo_estimates']:
            iterations, elos = zip(*self.metrics['elo_estimates'])
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, elos)
            plt.xlabel('Iteration')
            plt.ylabel('Estimated ELO')
            plt.title('ELO Rating Progress')
            plt.grid(True)
            
            # Save figure
            elo_plot_path = os.path.join(save_dir, 'elo_progress.png')
            plt.savefig(elo_plot_path)
            plt.close()
            
            self.log_info(f"ELO progress plot saved to {elo_plot_path}")


class Profiler:
    """
    Simple profiler for performance monitoring.
    
    This class provides timing utilities for profiling code execution.
    """
    
    def __init__(self):
        """Initialize the profiler."""
        self.timers = {}
        self.active_timers = {}
    
    def start(self, name):
        """
        Start a timer.
        
        Args:
            name: Timer name
        """
        self.active_timers[name] = time.time()
    
    def stop(self, name):
        """
        Stop a timer and record elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            elapsed: Elapsed time in seconds
        """
        if name not in self.active_timers:
            return 0
        
        elapsed = time.time() - self.active_timers[name]
        
        if name not in self.timers:
            self.timers[name] = []
        
        self.timers[name].append(elapsed)
        del self.active_timers[name]
        
        return elapsed
    
    def get_average(self, name):
        """
        Get average time for a timer.
        
        Args:
            name: Timer name
            
        Returns:
            average: Average time in seconds
        """
        if name not in self.timers or not self.timers[name]:
            return 0
        
        return sum(self.timers[name]) / len(self.timers[name])
    
    def get_total(self, name):
        """
        Get total time for a timer.
        
        Args:
            name: Timer name
            
        Returns:
            total: Total time in seconds
        """
        if name not in self.timers:
            return 0
        
        return sum(self.timers[name])
    
    def reset(self, name=None):
        """
        Reset timers.
        
        Args:
            name: Timer name to reset (None for all)
        """
        if name is None:
            self.timers = {}
            self.active_timers = {}
        elif name in self.timers:
            del self.timers[name]
            if name in self.active_timers:
                del self.active_timers[name]
    
    def summary(self):
        """
        Get a summary of all timers.
        
        Returns:
            summary: Dictionary of timer statistics
        """
        result = {}
        
        for name, times in self.timers.items():
            if not times:
                continue
                
            result[name] = {
                'count': len(times),
                'total': sum(times),
                'average': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        
        return result

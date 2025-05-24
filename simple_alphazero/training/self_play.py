"""
Self-play game generation for training.

This module handles the generation of self-play games for training the neural network.
"""

import chess
import numpy as np
import torch
import time
import random
from mcts.search import MCTS
from model.neural_net import board_to_tensor


class SelfPlay:
    """
    Self-play game generation for training data collection.

    This class handles:
    - Playing complete games using MCTS and the current neural network
    - Collecting states, policies, and outcomes for training
    - Applying temperature scheduling for exploration
    """

    def __init__(self, model, device='cuda', num_simulations=800,
                 max_moves=512, temperature_schedule=None):
        """
        Initialize the self-play generator.

        Args:
            model: Neural network model for position evaluation
            device: Device to run inference on ('cuda' or 'cpu')
            num_simulations: Number of MCTS simulations per move
            max_moves: Maximum number of moves per game
            temperature_schedule: Dictionary mapping move number to temperature
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.max_moves = max_moves

        # Default temperature schedule if none provided
        if temperature_schedule is None:
            self.temperature_schedule = {
                0: 1.0,    # Start with high temperature
                30: 0.5,   # Reduce after opening
                60: 0.25,  # Further reduce in middlegame
            }
        else:
            self.temperature_schedule = temperature_schedule

        # Create MCTS search
        self.mcts = MCTS(model, device, num_simulations)

    def play_game(self):
        """
        Play a complete game using MCTS and the current neural network.

        Returns:
            training_examples: List of (state, policy, value) tuples
            game_result: Final game result (1.0, 0.0, or -1.0)
        """
        board = chess.Board()
        training_examples = []

        move_count = 0
        states = []
        policies = []
        current_player = []

        while not board.is_game_over() and move_count < self.max_moves:
            temperature = self._get_temperature(move_count)

            # 🌟 NEW: Inject randomness in first few moves to break symmetry
            if move_count < 6:
                legal_moves = list(board.legal_moves)
                move = random.choice(legal_moves)
                policy = np.zeros(1968)  # 1968 is the policy size used in the neural network
                # For random moves, we'll just set a single position to 1.0
                # This is a simplified approach - in a full implementation, we'd map to proper move indices
                policy[0] = 1.0  # Set first element to 1.0 for random moves

                # Logging (optional):
                print(f"[RANDOM OPENING] Move {move_count + 1}: {board.san(move)}")

            else:
                # Standard MCTS move selection
                best_move, policy = self.mcts.search(board)
                move = best_move

            # Store current state and policy
            states.append(board.copy())
            policies.append(policy)
            current_player.append(board.turn)

            board.push(move)
            move_count += 1

            if self._should_terminate_early(board, move_count):
                print(f"[EARLY TERMINATION] Reason: {board.result()} at move {move_count}")
                break

        game_result = self._get_game_result(board)

        for i in range(len(states)):
            state_tensor = board_to_tensor(states[i])
            value = game_result if current_player[i] else -game_result
            training_examples.append((state_tensor, policies[i], value))

        return training_examples, game_result


    def _get_temperature(self, move_count):
        """
        Get temperature based on move count using the schedule.

        Args:
            move_count: Current move count

        Returns:
            temperature: Temperature value for current move
        """
        # Find the highest move threshold that's less than or equal to current move
        applicable_thresholds = [t for t in self.temperature_schedule.keys() if t <= move_count]

        if not applicable_thresholds:
            return 1.0  # Default temperature

        threshold = max(applicable_thresholds)
        return self.temperature_schedule[threshold]

    def _should_terminate_early(self, board, move_count):
        """
        Check if game should terminate early.

        Args:
            board: Current chess board
            move_count: Current move count

        Returns:
            bool: Whether to terminate the game early
        """
        if board.is_insufficient_material():
            print(f"[EARLY TERMINATION] Draw due to insufficient material at move {move_count}")
            return True

        if board.is_repetition(3):
            print(f"[EARLY TERMINATION] Draw due to threefold repetition at move {move_count}")
            return True

        if board.halfmove_clock >= 100:
            print(f"[EARLY TERMINATION] Draw due to 50-move rule at move {move_count}")
            return True

        if board.is_stalemate():
            print(f"[EARLY TERMINATION] Draw due to stalemate at move {move_count}")
            return True

        return False


    def _get_game_result(self, board):
        """
        Get the game result from white's perspective.

        Args:
            board: Final chess board

        Returns:
            result: Game result (1.0 for white win, -1.0 for black win, 0.0 for draw)
        """
        if board.is_checkmate():
            # If it's white's turn and checkmate, black won
            return -1.0 if board.turn else 1.0
        if board.is_repetition(3):
            return -0.2  # Slight penalty for repetition-based draw

        # All other cases are draws
        return 0.0

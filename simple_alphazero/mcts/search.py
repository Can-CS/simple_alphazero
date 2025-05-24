"""
MCTS search algorithm for chess.

This module implements the Monte Carlo Tree Search algorithm
for chess position evaluation and move selection.
"""

import numpy as np
import chess
import time
import torch
from .node import Node


class MCTS:
    """
    Monte Carlo Tree Search implementation for chess.

    This class handles the search process, integrating the neural network
    for position evaluation and prior move probabilities.
    """

    def __init__(self, model, device='cuda', num_simulations=800,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25, c_puct=1.0):
        """
        Initialize the MCTS search.

        Args:
            model: Neural network model for position evaluation
            device: Device to run inference on ('cuda' or 'cpu')
            num_simulations: Number of simulations to run per search
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Weight of Dirichlet noise to add
            c_puct: Exploration constant for UCB formula
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.c_puct = c_puct

    def search(self, board):
        """
        Perform MCTS search from the given position.

        Args:
            board: A chess.Board object representing the current position

        Returns:
            best_move: The selected best move
            move_probs: Probability distribution over moves
        """
        # Create root node
        root = Node(board)

        # Add Dirichlet noise to root node for exploration
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection - Find leaf node by repeatedly selecting best child
            while node.is_fully_expanded() and not node.is_terminal():
                node, move = node.select_child(self.c_puct)
                search_path.append(node)

            # Expansion and Evaluation
            if not node.is_terminal():
                # Get policy and value from neural network
                policy, value = self._evaluate_position(node.board)

                # Expand with all legal moves
                for move in node.untried_moves:
                    # Make the move on a new board
                    child_board = node.board.copy()
                    child_board.push(move)

                    # Get prior probability for this move
                    move_idx = self._move_to_index(move)
                    prior_prob = policy[move_idx] if move_idx < len(policy) else 0.001

                    # Create child node
                    node.expand(move, child_board, prior_prob)
            else:
                # Use terminal state result
                value = node.get_outcome()
                # Convert to current player's perspective
                if not node.board.turn:  # If it's black's turn
                    value = -value

            # Backpropagation
            for node in reversed(search_path):
                node.backpropagate(value)
                # Flip value for opponent's perspective
                value = -value

        # Calculate move probabilities based on visit counts
        move_probs = np.zeros(1968)  # Max possible moves in chess
        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits > 0:
            for move, child in root.children.items():
                move_idx = self._move_to_index(move)
                if move_idx < len(move_probs):
                    move_probs[move_idx] = child.visit_count / total_visits

        # Select best move (highest visit count)
        best_move = None
        best_count = -1

        for move, child in root.children.items():
            if child.visit_count > best_count:
                best_count = child.visit_count
                best_move = move

        return best_move, move_probs

    def _evaluate_position(self, board):
        """
        Evaluate a position using the neural network.

        Args:
            board: A chess.Board object

        Returns:
            policy: Probability distribution over moves
            value: Estimated position value
        """
        # Convert board to tensor representation
        from model.neural_net import board_to_tensor

        # Create tensor with contiguous memory layout for better GPU performance
        x = board_to_tensor(board).unsqueeze(0).contiguous().to(self.device, non_blocking=True)

        # Get policy and value from neural network with CUDA optimizations
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=True):
            # Use autocast for mixed precision inference
            policy_logits, value = self.model(x)

            # Convert policy logits to probabilities efficiently
            # Keep computation on GPU as long as possible
            policy = torch.softmax(policy_logits, dim=1).squeeze(0)

            # Only transfer to CPU at the end
            policy = policy.cpu().numpy()
            value = value.item()

        return policy, value

    def _add_dirichlet_noise(self, node):
        """
        Add Dirichlet noise to the prior probabilities at the root node.

        Args:
            node: Root node of the search tree
        """
        # Only add noise if node has children
        if not node.children:
            return

        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))

        # Apply noise to each child's prior probability
        for i, (move, child) in enumerate(node.children.items()):
            child.prior_prob = (1 - self.dirichlet_epsilon) * child.prior_prob + self.dirichlet_epsilon * noise[i]

    def _move_to_index(self, move):
        """
        Convert a chess move to an index in the policy vector.

        Args:
            move: A chess.Move object

        Returns:
            index: The corresponding index in the policy vector
        """
        # Simple encoding: from_square * 64 + to_square
        # This handles most moves but not promotions
        from_square = move.from_square
        to_square = move.to_square

        # Basic move encoding
        move_idx = from_square * 64 + to_square

        # Handle promotions (simplified)
        if move.promotion:
            # Add offset based on promotion piece type
            # Assuming policy vector has space for these
            promotion_offset = {
                chess.QUEEN: 0,
                chess.ROOK: 1,
                chess.BISHOP: 2,
                chess.KNIGHT: 3
            }
            move_idx = 4096 + from_square * 64 + to_square + promotion_offset.get(move.promotion, 0)

        return min(move_idx, 1967)  # Ensure index is within bounds

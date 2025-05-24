"""
MCTS node implementation for chess.

This module implements the Node class for Monte Carlo Tree Search,
representing positions in the chess game tree.
"""

import math
import numpy as np
import chess


class Node:
    """
    A node in the MCTS tree representing a chess position.
    
    Each node tracks:
    - The chess position
    - Statistics for the UCB formula (visits, value)
    - Children nodes and their prior probabilities
    - Parent node for backpropagation
    """
    
    def __init__(self, board, parent=None, prior_prob=0, move=None):
        """
        Initialize a new node.
        
        Args:
            board: A chess.Board object representing the position
            parent: The parent Node object (None for root)
            prior_prob: Prior probability of this node from the policy network
            move: The chess.Move that led to this position
        """
        self.board = board.copy()  # Copy to avoid reference issues
        self.parent = parent
        self.prior_prob = prior_prob
        self.move = move
        
        # Node statistics
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}  # Maps moves to child nodes
        
        # Unexpanded moves
        self.untried_moves = list(board.legal_moves)
        
    def is_fully_expanded(self):
        """Check if all legal moves have been expanded."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """Check if the position is a terminal state."""
        return self.board.is_game_over()
    
    def get_outcome(self):
        """Get the game outcome from this position."""
        if not self.is_terminal():
            return None
            
        result = self.board.result()
        if result == "1-0":
            return 1.0  # White wins
        elif result == "0-1":
            return -1.0  # Black wins
        else:
            return 0.0  # Draw
    
    def get_value(self):
        """Get the average value of this node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct=1.0):
        """
        Select a child node using the PUCT formula.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            The selected child node
        """
        # Find child with highest UCB score
        best_score = float('-inf')
        best_child = None
        best_move = None
        
        # Sum of all visit counts for normalization
        sum_visits = sum(child.visit_count for child in self.children.values())
        sum_visits = max(sum_visits, 1)  # Avoid division by zero
        
        for move, child in self.children.items():
            # UCB score = Q(s,a) + c_puct * P(s,a) * sqrt(sum(N(s,b))) / (1 + N(s,a))
            # Where:
            # - Q(s,a) is the mean value of the child
            # - P(s,a) is the prior probability from the policy network
            # - N(s,a) is the visit count of the child
            # - sum(N(s,b)) is the sum of all visit counts of all children
            
            # Exploitation term
            q_value = child.get_value()
            
            # For chess, flip the sign based on whose turn it is
            if not self.board.turn:  # If it's black's turn
                q_value = -q_value
                
            # Exploration term
            u_value = c_puct * child.prior_prob * math.sqrt(sum_visits) / (1 + child.visit_count)
            
            # Combined score
            ucb_score = q_value + u_value
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                best_move = move
                
        return best_child, best_move
    
    def expand(self, move, child_board, prior_prob):
        """
        Expand the tree by adding a new child node.
        
        Args:
            move: The chess move leading to the new position
            child_board: The resulting chess board
            prior_prob: Prior probability from policy network
            
        Returns:
            The newly created child node
        """
        child = Node(
            board=child_board,
            parent=self,
            prior_prob=prior_prob,
            move=move
        )
        
        self.children[move] = child
        
        # Remove this move from untried moves
        if move in self.untried_moves:
            self.untried_moves.remove(move)
            
        return child
    
    def backpropagate(self, value):
        """
        Update node statistics by backpropagating a value.
        
        Args:
            value: The value to backpropagate
        """
        # Update this node
        self.visit_count += 1
        self.value_sum += value
        
        # Recursively update parent nodes
        if self.parent:
            # Flip value for parent (opponent's perspective)
            self.parent.backpropagate(-value)

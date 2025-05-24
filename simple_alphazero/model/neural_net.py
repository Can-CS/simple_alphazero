"""
Lightweight neural network for chess position evaluation.

This module implements a simple convolutional neural network with dual policy and value heads,
inspired by AlphaZero but significantly simplified for faster training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChessNet(nn.Module):
    """
    A lightweight neural network for chess position evaluation.

    Architecture:
    - Input: 16-channel 8x8 board representation
    - 3 convolutional layers with batch normalization
    - Dual policy and value heads

    The network is designed to be fast and efficient while still capturing
    essential chess patterns for the 800-1400 ELO range.
    """

    def __init__(self, num_channels=64):
        """
        Initialize the neural network.

        Args:
            num_channels: Number of channels in convolutional layers
        """
        super(ChessNet, self).__init__()

        # Input layer
        self.conv1 = nn.Conv2d(16, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Hidden layers
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 1968)  # 1968 possible moves in chess

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor representing board state (batch_size, 16, 8, 8)

        Returns:
            policy_logits: Policy logits (batch_size, 1968)
            value: Value prediction (batch_size, 1)
        """
        # Input layer
        x = F.relu(self.bn1(self.conv1(x)))

        # Hidden layers
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value


def create_model(device='cuda'):
    """
    Create and initialize a new model.

    Args:
        device: Device to place the model on ('cuda' or 'cpu')

    Returns:
        model: Initialized ChessNet model
    """
    import sys
    sys.path.append('..')
    import config

    model = ChessNet(num_channels=64)
    model = model.to(device)

    # Apply torch.compile if enabled in config and running on CUDA
    if (device.type == 'cuda' if isinstance(device, torch.device) else device == 'cuda') and \
       config.RTX4070_OPTIMIZATIONS.get('use_torch_compile', False):
        compile_backend = config.RTX4070_OPTIMIZATIONS.get('compile_backend', 'inductor')
        compile_mode = config.RTX4070_OPTIMIZATIONS.get('compile_mode', 'default')

        print(f"Applying torch.compile with mode: {compile_mode}, backend: {compile_backend}")
        model = torch.compile(model, mode=compile_mode, backend=compile_backend)

    return model


def board_to_tensor(board):
    """
    Convert a chess board to a tensor representation.

    Args:
        board: A chess.Board object

    Returns:
        tensor: A 16x8x8 tensor representing the board state
    """
    # Initialize tensor
    tensor = torch.zeros(16, 8, 8)

    # Piece channels (6 piece types x 2 colors = 12 channels)
    piece_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }

    # Fill piece channels
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            row, col = square // 8, square % 8
            tensor[piece_idx[piece.symbol()]][row][col] = 1

    # Additional features
    # Channel 12: All ones if white to move, all zeros if black to move
    tensor[12].fill_(1 if board.turn else 0)

    # Channel 13: Castling rights
    for square in range(64):
        row, col = square // 8, square % 8
        # White kingside
        if board.has_kingside_castling_rights(True):
            tensor[13][7][4] = 1
            tensor[13][7][7] = 1
        # White queenside
        if board.has_queenside_castling_rights(True):
            tensor[13][7][4] = 1
            tensor[13][7][0] = 1
        # Black kingside
        if board.has_kingside_castling_rights(False):
            tensor[13][0][4] = 1
            tensor[13][0][7] = 1
        # Black queenside
        if board.has_queenside_castling_rights(False):
            tensor[13][0][4] = 1
            tensor[13][0][0] = 1

    # Channel 14: En passant
    if board.ep_square is not None:
        row, col = board.ep_square // 8, board.ep_square % 8
        tensor[14][row][col] = 1

    # Channel 15: Move count (normalized)
    tensor[15].fill_(min(1.0, board.fullmove_number / 100.0))

    return tensor

"""
Chess GUI for visualization and interaction.

This module provides a simple GUI for visualizing chess games,
allowing users to play against the AI or view self-play games.
"""

import pygame
import chess
import chess.svg
import cairosvg
import io
import numpy as np
from PIL import Image


class ChessGUI:
    """
    A simple chess GUI using Pygame.
    
    This class handles:
    - Rendering the chess board and pieces
    - Handling user input for moves
    - Displaying game information
    """
    
    def __init__(self, width=600, height=600):
        """
        Initialize the chess GUI.
        
        Args:
            width: Width of the window
            height: Height of the window
        """
        # Initialize Pygame
        pygame.init()
        
        # Set up the display
        self.width = width
        self.height = height
        self.board_size = min(width, height)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Simple AlphaZero Chess")
        
        # Initialize chess board
        self.board = chess.Board()
        
        # Track selected square
        self.selected_square = None
        
        # Colors
        self.colors = {
            'background': (240, 240, 240),
            'text': (0, 0, 0),
            'highlight': (255, 255, 0, 128),  # Yellow with alpha
            'last_move': (0, 255, 0, 128),    # Green with alpha
        }
        
        # Font
        self.font = pygame.font.SysFont('Arial', 16)
        
    def run(self, ai_player=None):
        """
        Run the GUI main loop.
        
        Args:
            ai_player: Optional AI player function that takes a board and returns a move
        """
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle mouse click
                    self._handle_click(event.pos)
            
            # If it's AI's turn
            if ai_player and ((self.board.turn == chess.WHITE and ai_player.color == chess.WHITE) or
                             (self.board.turn == chess.BLACK and ai_player.color == chess.BLACK)):
                move = ai_player.get_move(self.board)
                if move:
                    self.board.push(move)
            
            # Draw the board
            self._draw_board()
            
            # Update the display
            pygame.display.flip()
            
            # Cap the frame rate
            clock.tick(30)
        
        pygame.quit()
    
    def _handle_click(self, pos):
        """
        Handle mouse click on the board.
        
        Args:
            pos: Mouse position (x, y)
        """
        # Convert mouse position to board square
        x, y = pos
        square_size = self.board_size / 8
        file = int(x / square_size)
        rank = 7 - int(y / square_size)  # Flip rank (0 is bottom in chess)
        
        # Ensure click is within board
        if 0 <= file < 8 and 0 <= rank < 8:
            square = chess.square(file, rank)
            
            # If no square is selected, select this one
            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
            else:
                # Try to make a move
                move = chess.Move(self.selected_square, square)
                
                # Check for promotion
                if self.board.piece_at(self.selected_square) and \
                   self.board.piece_at(self.selected_square).piece_type == chess.PAWN and \
                   (rank == 0 or rank == 7):
                    # Promote to queen (simplified)
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                
                # Make the move if legal
                if move in self.board.legal_moves:
                    self.board.push(move)
                
                # Clear selection
                self.selected_square = None
    
    def _draw_board(self):
        """Draw the chess board and pieces."""
        # Clear the screen
        self.screen.fill(self.colors['background'])
        
        # Get SVG representation of the board
        svg_data = chess.svg.board(
            board=self.board,
            size=self.board_size,
            lastmove=self.board.move_stack[-1] if self.board.move_stack else None,
            check=self.board.king(self.board.turn) if self.board.is_check() else None,
            squares=chess.SquareSet([self.selected_square]) if self.selected_square is not None else None
        )
        
        # Convert SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
        
        # Convert PNG to Pygame surface
        image = Image.open(io.BytesIO(png_data))
        mode = image.mode
        size = image.size
        data = image.tobytes()
        
        # Create Pygame surface
        board_surface = pygame.image.fromstring(data, size, mode)
        
        # Draw the board
        self.screen.blit(board_surface, (0, 0))
        
        # Draw game status
        self._draw_status()
    
    def _draw_status(self):
        """Draw game status information."""
        # Create status text
        if self.board.is_checkmate():
            status = "Checkmate! " + ("Black" if self.board.turn == chess.WHITE else "White") + " wins"
        elif self.board.is_stalemate():
            status = "Stalemate! Draw"
        elif self.board.is_insufficient_material():
            status = "Insufficient material! Draw"
        elif self.board.is_check():
            status = "Check! " + ("White" if self.board.turn == chess.WHITE else "Black") + " to move"
        else:
            status = "White to move" if self.board.turn == chess.WHITE else "Black to move"
        
        # Render text
        text_surface = self.font.render(status, True, self.colors['text'])
        
        # Position text at the bottom of the screen
        text_rect = text_surface.get_rect(center=(self.width/2, self.height - 20))
        
        # Draw text
        self.screen.blit(text_surface, text_rect)


class AIPlayer:
    """
    AI player using MCTS for move selection.
    
    This class provides an interface for the GUI to get moves from the AI.
    """
    
    def __init__(self, mcts, color=chess.BLACK):
        """
        Initialize the AI player.
        
        Args:
            mcts: MCTS search object
            color: Color the AI plays as (chess.WHITE or chess.BLACK)
        """
        self.mcts = mcts
        self.color = color
    
    def get_move(self, board):
        """
        Get the best move for the current position.
        
        Args:
            board: Current chess board
            
        Returns:
            move: Selected chess move
        """
        # Only make a move if it's our turn
        if board.turn != self.color:
            return None
        
        # Use MCTS to find the best move
        best_move, _ = self.mcts.search(board)
        return best_move

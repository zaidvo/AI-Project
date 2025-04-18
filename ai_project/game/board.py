# game/board.py
import numpy as np
from .constants import EMPTY, PLAYER_X, PLAYER_O, BOARD_SIZE, WIN_LENGTH

class Board:
    def __init__(self, size=BOARD_SIZE):
        """
        Initialize game board
        
        Args:
            size (int): Size of board (default: 7)
        """
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.last_move = None
        self.current_player = PLAYER_X  # X always starts
        
    def reset(self):
        """Reset board to initial state"""
        self.grid.fill(EMPTY)
        self.last_move = None
        self.current_player = PLAYER_X
        
    def get_valid_moves(self):
        """
        Get all valid moves (empty cells)
        
        Returns:
            list: Indices of valid moves (flattened)
        """
        return np.where(self.grid.flatten() == EMPTY)[0].tolist()
    
    def make_move(self, position):
        """
        Make a move at given position
        
        Args:
            position (int): Flattened position (0-48)
            
        Returns:
            bool: True if move was valid and made, False otherwise
        """
        # Convert flattened position to 2D coordinates
        row, col = divmod(position, self.size)
        
        # Check if move is valid
        if not self.is_valid_move(position):
            return False
        
        # Make move
        self.grid[row, col] = self.current_player
        self.last_move = (row, col)
        
        # Switch player
        self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
        
        return True
    
    def is_valid_move(self, position):
        """
        Check if move is valid
        
        Args:
            position (int): Flattened position
            
        Returns:
            bool: True if move is valid, False otherwise
        """
        if position < 0 or position >= self.size * self.size:
            return False
            
        row, col = divmod(position, self.size)
        return self.grid[row, col] == EMPTY
    
    def check_win(self):
        """
        Check if there's a winner
        
        Returns:
            int: Winner (PLAYER_X or PLAYER_O) or None
        """
        if self.last_move is None:
            return None
            
        last_row, last_col = self.last_move
        last_player = self.grid[last_row, last_col]
        
        # Define directions to check (horizontal, vertical, diagonals)
        directions = [
            [(0, 1), (0, -1)],   # horizontal
            [(1, 0), (-1, 0)],   # vertical
            [(1, 1), (-1, -1)],  # diagonal \
            [(1, -1), (-1, 1)]   # diagonal /
        ]
        
        # Check each direction
        for dir_pair in directions:
            count = 1  # Start with 1 for the last move
            
            # Check both directions
            for dr, dc in dir_pair:
                r, c = last_row, last_col
                
                # Count consecutive pieces
                for _ in range(WIN_LENGTH - 1):
                    r, c = r + dr, c + dc
                    if 0 <= r < self.size and 0 <= c < self.size and self.grid[r, c] == last_player:
                        count += 1
                    else:
                        break
            
            # Check if we have enough consecutive pieces
            if count >= WIN_LENGTH:
                return last_player
                
        return None
    
    def is_draw(self):
        """
        Check if game ended in a draw
        
        Returns:
            bool: True if draw, False otherwise
        """
        return len(self.get_valid_moves()) == 0 and self.check_win() is None
    
    def is_game_over(self):
        """
        Check if game is over
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.check_win() is not None or self.is_draw()
    
    def get_state(self):
        """
        Get current state as numpy array
        
        Returns:
            np.array: Flattened board state
        """
        return self.grid.flatten().copy()
    
    def __str__(self):
        """String representation of board"""
        symbols = {EMPTY: '.', PLAYER_X: 'X', PLAYER_O: 'O'}
        result = ""
        
        # Add column indices
        result += "  " + " ".join(str(i) for i in range(self.size)) + "\n"
        
        # Add rows
        for i, row in enumerate(self.grid):
            result += f"{i} " + " ".join(symbols[cell] for cell in row) + "\n"
            
        return result
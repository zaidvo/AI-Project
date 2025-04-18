# gui/renderer.py
import pygame
import pygame.gfxdraw
from game.constants import *

class GameRenderer:
    def __init__(self, screen):
        """
        Initialize game renderer
        
        Args:
            screen: Pygame screen object
        """
        self.screen = screen
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        
    def render(self, board, game_over, winner):
        """
        Render the game board and status
        
        Args:
            board: Game board object
            game_over: Whether game is over
            winner: Winner (PLAYER_X, PLAYER_O, or None for draw)
        """
        # Clear screen
        self.screen.fill(BG_COLOR)
        
        # Draw board grid
        self._draw_grid()
        
        # Draw X's and O's
        self._draw_pieces(board)
        
        # Draw game status
        self._draw_status(board, game_over, winner)
        
    def _draw_grid(self):
        """Draw the game grid"""
        for i in range(BOARD_SIZE + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (BOARD_PADDING + i * CELL_SIZE, BOARD_PADDING),
                (BOARD_PADDING + i * CELL_SIZE, BOARD_PADDING + BOARD_SIZE * CELL_SIZE),
                2
            )
            
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (BOARD_PADDING, BOARD_PADDING + i * CELL_SIZE),
                (BOARD_PADDING + BOARD_SIZE * CELL_SIZE, BOARD_PADDING + i * CELL_SIZE),
                2
            )
    
    def _draw_pieces(self, board):
        """
        Draw X's and O's on the board
        
        Args:
            board: Game board object
        """
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell_value = board.grid[row][col]
                if cell_value == PLAYER_X:
                    self._draw_x(row, col)
                elif cell_value == PLAYER_O:
                    self._draw_o(row, col)
    
    def _draw_x(self, row, col):
        """Draw X at given position"""
        x = BOARD_PADDING + col * CELL_SIZE + CELL_SIZE // 2
        y = BOARD_PADDING + row * CELL_SIZE + CELL_SIZE // 2
        size = CELL_SIZE // 3
        
        # Draw X as two lines
        pygame.draw.line(
            self.screen,
            X_COLOR,
            (x - size, y - size),
            (x + size, y + size),
            6
        )
        pygame.draw.line(
            self.screen,
            X_COLOR,
            (x + size, y - size),
            (x - size, y + size),
            6
        )
    
    def _draw_o(self, row, col):
        """Draw O at given position"""
        x = BOARD_PADDING + col * CELL_SIZE + CELL_SIZE // 2
        y = BOARD_PADDING + row * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 3
        
        # Draw O as a circle
        pygame.draw.circle(
            self.screen,
            O_COLOR,
            (x, y),
            radius,
            4
        )
    
    def _draw_status(self, board, game_over, winner):
        """
        Draw game status text
        
        Args:
            board: Game board object
            game_over: Whether game is over
            winner: Winner (PLAYER_X, PLAYER_O, or None for draw)
        """
        status_text = ""
        
        if game_over:
            if winner == PLAYER_X:
                status_text = "X Wins!"
            elif winner == PLAYER_O:
                status_text = "O Wins!"
            else:
                status_text = "Draw!"
        else:
            status_text = "X's Turn" if board.current_player == PLAYER_X else "O's Turn"
        
        # Render status text
        text_surface = self.font.render(status_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(WIDTH // 2, BOARD_PADDING + BOARD_SIZE * CELL_SIZE + 30))
        self.screen.blit(text_surface, text_rect)
    
    def get_cell_from_pos(self, pos):
        """
        Get board cell from mouse position
        
        Args:
            pos: Mouse position (x, y)
            
        Returns:
            tuple: (row, col) or None if outside board
        """
        x, y = pos
        
        # Check if click is inside board
        if (x < BOARD_PADDING or x > BOARD_PADDING + BOARD_SIZE * CELL_SIZE or
            y < BOARD_PADDING or y > BOARD_PADDING + BOARD_SIZE * CELL_SIZE):
            return None
        
        # Calculate cell coordinates
        col = (x - BOARD_PADDING) // CELL_SIZE
        row = (y - BOARD_PADDING) // CELL_SIZE
        
        return (row, col)
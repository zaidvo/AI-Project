# models/main.py
import os
import sys
import pygame
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from game.constants import *
from ai.dqn_agent import DQNAgent
from gui.renderer import GameRenderer
from gui.button import Button

class Game:
    def __init__(self):
        """Initialize game"""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MegaTicTacToe - 7x7")
        
        self.board = Board()
        self.renderer = GameRenderer(self.screen)
        
        # Load DQN agent
        self.agent = DQNAgent()
        model_path = os.path.join(os.path.dirname(__file__), "dqn_megatictactoe.h5")
        if os.path.exists(model_path):
            self.agent.load(model_path)
            self.agent.epsilon = 0.01  # Low exploration during gameplay
        
        # Create buttons
        self.reset_button = Button(
            WIDTH // 2 - BUTTON_WIDTH - 10,
            HEIGHT - BUTTON_HEIGHT - 20,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "New Game",
            self.reset_game
        )
        
        self.ai_first_button = Button(
            WIDTH // 2 + 10,
            HEIGHT - BUTTON_HEIGHT - 20,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "AI First",
            self.ai_first_move
        )
        
        self.game_over = False
        self.winner = None
        self.player_turn = True  # Human player starts by default
        
    def reset_game(self):
        """Reset game state"""
        self.board.reset()
        self.game_over = False
        self.winner = None
        self.player_turn = True
        
    def ai_first_move(self):
        """Let AI make first move"""
        if not self.game_over and self.player_turn and len(self.board.get_valid_moves()) == self.board.size ** 2:
            self.player_turn = False
            self.make_ai_move()
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Handle mouse clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check if clicked on board
                if self.player_turn and not self.game_over:
                    pos = pygame.mouse.get_pos()
                    cell = self.renderer.get_cell_from_pos(pos)
                    if cell is not None:
                        row, col = cell
                        position = row * BOARD_SIZE + col
                        if self.board.is_valid_move(position):
                            self.board.make_move(position)
                            self.check_game_state()
                            self.player_turn = False
                
                # Check button clicks
                self.reset_button.check_click(pygame.mouse.get_pos())
                self.ai_first_button.check_click(pygame.mouse.get_pos())
                
        return True
    
    def make_ai_move(self):
        """Make AI move"""
        if self.game_over or self.player_turn:
            return
            
        # Get current state
        state = self.board.get_state()
        
        # Get valid moves
        valid_moves = self.board.get_valid_moves()
        
        # Choose action
        action = self.agent.act(state, valid_moves, training=False)
        
        # Make move
        if action is not None:
            self.board.make_move(action)
            self.check_game_state()
            self.player_turn = True
    
    def check_game_state(self):
        """Check if game is over"""
        winner = self.board.check_win()
        if winner is not None:
            self.game_over = True
            self.winner = winner
        elif self.board.is_draw():
            self.game_over = True
            self.winner = None
    
    def run(self):
        """Main game loop"""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Make AI move if it's AI's turn
            if not self.player_turn and not self.game_over:
                self.make_ai_move()
            
            # Render game
            self.renderer.render(self.board, self.game_over, self.winner)
            self.reset_button.draw(self.screen)
            self.ai_first_button.draw(self.screen)
            
            # Update display
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
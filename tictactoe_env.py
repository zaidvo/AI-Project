import numpy as np
from typing import Tuple, Optional


class TicTacToeEnv:
    """
    7x7 Tic Tac Toe Environment
    Player 1: X (represented as 1)
    Player 2: O (represented as -1)
    Empty: 0

    The goal is to get 4 in a row (horizontally, vertically, or diagonally)
    """

    def __init__(self):
        self.size = 7
        self.win_length = 4  # Number in a row needed to win
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1  # X starts
        self.done = False
        self.winner = None

    def reset(self) -> np.ndarray:
        """Reset the game to initial state and return the board"""
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return the current state representation"""
        return self.board.copy()

    def get_valid_moves(self) -> list:
        """Return list of valid move indices (0-48)"""
        if self.done:
            return []

        valid = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    valid.append(i * self.size + j)
        return valid

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return new state, reward, done, info

        Args:
            action: Integer action from 0 to 48 representing position on board

        Returns:
            state: New board state
            reward: Reward for the action
            done: Whether game is finished
            info: Additional information dictionary
        """
        if self.done:
            return self.get_state(), 0, True, {"winner": self.winner}

        # Convert action to row, col
        row, col = action // self.size, action % self.size

        # Check if valid move
        if self.board[row, col] != 0:
            # Invalid move, penalize and don't change state
            return self.get_state(), -10, False, {"valid": False}

        # Make move
        self.board[row, col] = self.current_player

        # Check for win
        winner = self._check_winner()
        if winner is not None:
            self.done = True
            self.winner = winner
            reward = 1.0 if winner == self.current_player else -1.0
            return self.get_state(), reward, True, {"winner": winner}

        # Check for draw
        if np.all(self.board != 0):
            self.done = True
            self.winner = 0  # Draw
            return self.get_state(), 0.1, True, {"winner": 0}

        # Switch player
        self.current_player = -self.current_player

        return self.get_state(), 0.0, False, {}

    def _check_winner(self) -> Optional[int]:
        """Check if there's a winner and return the player number or None"""
        # Check rows
        for i in range(self.size):
            for j in range(self.size - self.win_length + 1):
                window = self.board[i, j : j + self.win_length]
                if np.all(window == 1):
                    return 1
                if np.all(window == -1):
                    return -1

        # Check columns
        for i in range(self.size - self.win_length + 1):
            for j in range(self.size):
                window = self.board[i : i + self.win_length, j]
                if np.all(window == 1):
                    return 1
                if np.all(window == -1):
                    return -1

        # Check diagonals (top-left to bottom-right)
        for i in range(self.size - self.win_length + 1):
            for j in range(self.size - self.win_length + 1):
                window = [self.board[i + k, j + k] for k in range(self.win_length)]
                if all(cell == 1 for cell in window):
                    return 1
                if all(cell == -1 for cell in window):
                    return -1

        # Check diagonals (top-right to bottom-left)
        for i in range(self.size - self.win_length + 1):
            for j in range(self.win_length - 1, self.size):
                window = [self.board[i + k, j - k] for k in range(self.win_length)]
                if all(cell == 1 for cell in window):
                    return 1
                if all(cell == -1 for cell in window):
                    return -1

        # No winner
        return None

    def render(self):
        """Print the board state"""
        symbols = {0: ".", 1: "X", -1: "O"}
        for i in range(self.size):
            row = [symbols[x] for x in self.board[i]]
            print(" ".join(row))
        print()

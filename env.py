import numpy as np


class TicTacToeEnv:
    def __init__(self, board_size=5, win_length=4):
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.last_move = None
        return self.get_observation()

    def get_observation(self):
        """Returns a 2-channel observation (current player pieces, opponent pieces)"""
        observation = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
        observation[0] = self.board == 1
        observation[1] = self.board == -1
        return observation

    def get_valid_moves(self):
        """Returns a matrix of valid moves"""
        if self.done:
            return np.zeros((self.board_size, self.board_size), dtype=np.int8)
        return (self.board == 0).astype(np.int8)

    def get_valid_moves_flat(self):
        """Returns a flattened array of valid moves"""
        return self.get_valid_moves().flatten()

    def step(self, action):
        """Take a step with action, return (observation, reward, done, info)"""
        if isinstance(action, (int, np.int32, np.int64)):
            x, y = action // self.board_size, action % self.board_size
        else:
            x, y = action

        # Invalid move
        if self.board[x, y] != 0 or self.done:
            return self.get_observation(), -10, True, {"winner": self.winner}

        # Make move
        self.board[x, y] = self.current_player
        self.last_move = (x, y)

        # Check for win
        if self._check_win(x, y):
            self.done = True
            self.winner = self.current_player
            return self.get_observation(), 1, True, {"winner": self.winner}

        # Check for draw
        if np.all(self.board != 0):
            self.done = True
            self.winner = 0
            return self.get_observation(), 0, True, {"winner": self.winner}

        # Switch player
        self.current_player *= -1

        return self.get_observation(), 0, False, {"winner": None}

    def _check_win(self, x, y):
        """Check if the last move at (x, y) created a winning line"""
        player = self.board[x, y]

        # Check horizontally
        for c in range(
            max(0, y - self.win_length + 1),
            min(y + 1, self.board_size - self.win_length + 1),
        ):
            if np.all(self.board[x, c : c + self.win_length] == player):
                return True

        # Check vertically
        for r in range(
            max(0, x - self.win_length + 1),
            min(x + 1, self.board_size - self.win_length + 1),
        ):
            if np.all(self.board[r : r + self.win_length, y] == player):
                return True

        # Check diagonal (top-left to bottom-right)
        for i in range(
            -min(x, y, self.win_length - 1),
            min(self.board_size - x, self.board_size - y, self.win_length),
        ):
            diag = []
            for j in range(self.win_length):
                if (
                    0 <= x + i + j < self.board_size
                    and 0 <= y + i + j < self.board_size
                ):
                    diag.append(self.board[x + i + j, y + i + j])
                else:
                    break
            if len(diag) >= self.win_length and np.all(np.array(diag) == player):
                return True

        # Check diagonal (top-right to bottom-left)
        for i in range(
            -min(x, self.board_size - 1 - y, self.win_length - 1),
            min(self.board_size - x, y + 1, self.win_length),
        ):
            diag = []
            for j in range(self.win_length):
                if (
                    0 <= x + i + j < self.board_size
                    and 0 <= y - i - j < self.board_size
                ):
                    diag.append(self.board[x + i + j, y - i - j])
                else:
                    break
            if len(diag) >= self.win_length and np.all(np.array(diag) == player):
                return True

        return False

    def render(self):
        """Print the current board state"""
        symbols = {0: ".", 1: "X", -1: "O"}
        print("  " + " ".join(str(i) for i in range(self.board_size)))
        for i in range(self.board_size):
            row = symbols[self.board[i, 0]]
            for j in range(1, self.board_size):
                row += " " + symbols[self.board[i, j]]
            print(f"{i} {row}")
        print()

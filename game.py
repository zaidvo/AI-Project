from config import *


class TicTacToe:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = [[EMPTY for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = PLAYER_X

    def get_valid_moves(self):
        return [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r][c] == EMPTY
        ]

    def make_move(self, move, player):
        r, c = move
        if self.board[r][c] == EMPTY:
            self.board[r][c] = player
            return True
        return False

    def undo_move(self, move):
        r, c = move
        self.board[r][c] = EMPTY

    def is_draw(self):
        return (
            not self.check_winner(PLAYER_X)
            and not self.check_winner(PLAYER_O)
            and not self.get_valid_moves()
        )

    def is_game_over(self):
        return (
            self.check_winner(PLAYER_X) or self.check_winner(PLAYER_O) or self.is_draw()
        )

    def check_winner(self, player):
        lines = self.get_all_lines()
        for line in lines:
            for i in range(len(line) - WIN_LENGTH + 1):
                if all(cell == player for cell in line[i : i + WIN_LENGTH]):
                    return True
        return False

    def get_all_lines(self):
        lines = []

        # Rows and Columns
        for i in range(self.board_size):
            lines.append(self.board[i])  # Row
            lines.append([self.board[j][i] for j in range(self.board_size)])  # Column

        # Diagonals: top-left to bottom-right
        for r in range(self.board_size - WIN_LENGTH + 1):
            for c in range(self.board_size - WIN_LENGTH + 1):
                diag1 = [
                    self.board[r + i][c + i]
                    for i in range(min(self.board_size - r, self.board_size - c))
                ]
                if len(diag1) >= WIN_LENGTH:
                    lines.append(diag1)

        # Diagonals: top-right to bottom-left
        for r in range(self.board_size - WIN_LENGTH + 1):
            for c in range(WIN_LENGTH - 1, self.board_size):
                diag2 = [
                    self.board[r + i][c - i]
                    for i in range(min(self.board_size - r, c + 1))
                ]
                if len(diag2) >= WIN_LENGTH:
                    lines.append(diag2)

        return lines

from config import *

# Scoring constants
FOUR_IN_ROW_SCORE = 10000
THREE_IN_ROW_SCORE = 100
TWO_IN_ROW_SCORE = 10
ONE_IN_ROW_SCORE = 1
BLOCK_WEIGHT = 1.2


class MinimaxAgent:
    def __init__(self, player, opponent, depth=4):
        self.player = player
        self.opponent = opponent
        self.depth = depth
        self.cache = {}

    def choose_move(self, game):
        self.cache.clear()  # Clear cache per move to limit memory use
        _, move = self.minimax(game, self.depth, float("-inf"), float("inf"), True)
        return move

    def minimax(self, game, depth, alpha, beta, maximizing):
        board_key = self._board_hash(game)

        if board_key in self.cache:
            return self.cache[board_key]

        if game.check_winner(self.player):
            return FOUR_IN_ROW_SCORE, None
        if game.check_winner(self.opponent):
            return -FOUR_IN_ROW_SCORE, None
        if not game.get_valid_moves() or depth == 0:
            eval_score = self.evaluate(game)
            self.cache[board_key] = (eval_score, None)
            return eval_score, None

        best_move = None
        moves = game.get_valid_moves()
        ordered_moves = self._order_moves(game, moves, maximizing)

        if maximizing:
            max_eval = float("-inf")
            for move in ordered_moves:
                game.make_move(move, self.player)
                eval, _ = self.minimax(game, depth - 1, alpha, beta, False)
                game.undo_move(move)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.cache[board_key] = (max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in ordered_moves:
                game.make_move(move, self.opponent)
                eval, _ = self.minimax(game, depth - 1, alpha, beta, True)
                game.undo_move(move)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.cache[board_key] = (min_eval, best_move)
            return min_eval, best_move

    def evaluate(self, game):
        def score_line(line, symbol):
            score = 0
            for i in range(len(line) - WIN_LENGTH + 1):
                window = line[i : i + WIN_LENGTH]
                count = window.count(symbol)
                empty = window.count(EMPTY)
                if count == 4 and empty == 0:
                    score += FOUR_IN_ROW_SCORE
                elif count == 3 and empty == 1:
                    score += THREE_IN_ROW_SCORE
                elif count == 2 and empty == 2:
                    score += TWO_IN_ROW_SCORE
                elif count == 1 and empty == 3:
                    score += ONE_IN_ROW_SCORE
            return score

        total = 0
        lines = game.get_all_lines()

        for line in lines:
            total += score_line(line, self.player)
            total -= score_line(line, self.opponent) * BLOCK_WEIGHT

        return total

    def _board_hash(self, game):
        # Immutable and hashable board representation
        return tuple(tuple(row) for row in game.board)

    def _order_moves(self, game, moves, maximizing):
        # Score moves using shallow evaluation (heuristic only)
        move_scores = []
        for move in moves:
            game.make_move(move, self.player if maximizing else self.opponent)
            score = self.evaluate(game)
            move_scores.append((score, move))
            game.undo_move(move)

        # Sort descending for maximizing, ascending for minimizing
        move_scores.sort(reverse=maximizing)
        return [move for _, move in move_scores]

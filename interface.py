import sys
import pygame
from copy import deepcopy

from config import *
from game import TicTacToe
from agent import MinimaxAgent

pygame.init()
font = pygame.font.SysFont(None, 60)
small_font = pygame.font.SysFont(None, 36)


def choose_from_options(title, options):
    screen = pygame.display.set_mode((CELL_SIZE * 7, CELL_SIZE * 7 + 100))
    screen.fill(WHITE)
    title_text = small_font.render(title, True, BLACK)
    screen.blit(title_text, (screen.get_width() // 2 - title_text.get_width() // 2, 40))

    rects = []
    for i, option in enumerate(options):
        rect = pygame.Rect(screen.get_width() // 2 - 100, 100 + i * 70, 200, 50)
        pygame.draw.rect(screen, GRAY, rect, border_radius=10)
        label = small_font.render(option, True, BLACK)
        screen.blit(label, label.get_rect(center=rect.center))
        rects.append((rect, option))
    pygame.display.flip()

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for rect, value in rects:
                    if rect.collidepoint(event.pos):
                        return value


def draw_board(screen, game, board_size, winning_line=None, hover_cell=None):
    screen.fill(WHITE)
    for r in range(1, board_size):
        pygame.draw.line(
            screen,
            GRAY,
            (0, r * CELL_SIZE),
            (board_size * CELL_SIZE, r * CELL_SIZE),
            LINE_WIDTH,
        )
        pygame.draw.line(
            screen,
            GRAY,
            (r * CELL_SIZE, 0),
            (r * CELL_SIZE, board_size * CELL_SIZE),
            LINE_WIDTH,
        )

    for r in range(board_size):
        for c in range(board_size):
            val = game.board[r][c]
            if val != EMPTY:
                color = RED if winning_line and (r, c) in winning_line else BLACK
                text = font.render(val, True, color)
                screen.blit(text, (c * CELL_SIZE + 30, r * CELL_SIZE + 20))
            elif hover_cell == (r, c):
                pygame.draw.rect(
                    screen,
                    (200, 200, 200),
                    (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                )


def show_message(screen, text, board_size):
    pygame.draw.rect(
        screen, WHITE, (0, board_size * CELL_SIZE, screen.get_width(), 100)
    )
    msg = small_font.render(text, True, BLACK)
    screen.blit(msg, (20, board_size * CELL_SIZE + 20))


def draw_play_again_button(screen, mouse_pos, board_size):
    rect = pygame.Rect(
        screen.get_width() // 2 - 80, board_size * CELL_SIZE + 50, 160, 40
    )
    pygame.draw.rect(
        screen,
        GRAY if not rect.collidepoint(mouse_pos) else (180, 180, 180),
        rect,
        border_radius=10,
    )
    text = small_font.render("Play Again", True, BLACK)
    screen.blit(text, text.get_rect(center=rect.center))
    return rect


def get_winning_line(game, board_size, player):
    for r in range(board_size):
        for c in range(board_size):
            directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
            for dr, dc in directions:
                line = []
                for i in range(WIN_LENGTH):
                    nr, nc = r + dr * i, c + dc * i
                    if (
                        0 <= nr < board_size
                        and 0 <= nc < board_size
                        and game.board[nr][nc] == player
                    ):
                        line.append((nr, nc))
                    else:
                        break
                if len(line) == WIN_LENGTH:
                    return line
    return None


def handle_human_move(game, event, board_size):
    if event.type == pygame.MOUSEBUTTONDOWN:
        x, y = event.pos
        r, c = y // CELL_SIZE, x // CELL_SIZE
        if r < board_size and c < board_size:
            return game.make_move((r, c), game.current_player)
    return False


def handle_ai_move(screen, game, agent, board_size):
    show_message(screen, "AI is thinking...", board_size)
    pygame.display.flip()
    pygame.time.delay(300)
    move = agent.choose_move(deepcopy(game))
    if move:
        return game.make_move(move, game.current_player)
    return False


def main():
    while True:
        grid_size = 5
        mode = choose_from_options(
            "Select Game Mode:", ["Human vs Human", "Human vs AI", "AI vs AI"]
        )
        depth = int(choose_from_options("Select AI Depth:", ["2", "3", "4", "5"]))
        player_first = True

        if mode == "Human vs AI":
            turn = choose_from_options("Do you want to go first?", ["Yes", "No"])
            player_first = turn == "Yes"

        screen = pygame.display.set_mode(
            (CELL_SIZE * grid_size, CELL_SIZE * grid_size + 100)
        )
        game = TicTacToe(grid_size)
        clock = pygame.time.Clock()

        agent_X = (
            MinimaxAgent(PLAYER_X, PLAYER_O, depth)
            if "AI" in mode and (mode == "AI vs AI" or not player_first)
            else None
        )
        agent_O = (
            MinimaxAgent(PLAYER_O, PLAYER_X, depth)
            if "AI" in mode and (mode == "AI vs AI" or player_first)
            else None
        )

        game.current_player = PLAYER_X if player_first else PLAYER_O
        game_over = False
        winning_line = None

        while True:
            mouse_pos = pygame.mouse.get_pos()
            hover_cell = (
                (mouse_pos[1] // CELL_SIZE, mouse_pos[0] // CELL_SIZE)
                if mouse_pos[1] < CELL_SIZE * grid_size
                else None
            )

            draw_board(screen, game, grid_size, winning_line, hover_cell)

            if not game_over:
                show_message(
                    screen, f"{game.current_player}'s turn - {mode}", grid_size
                )
                current_agent = agent_X if game.current_player == PLAYER_X else agent_O
                move_made = False

                if current_agent:
                    move_made = handle_ai_move(screen, game, current_agent, grid_size)
                else:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if handle_human_move(game, event, grid_size):
                            move_made = True

                if move_made:
                    if game.check_winner(game.current_player):
                        winning_line = get_winning_line(
                            game, grid_size, game.current_player
                        )
                        show_message(screen, f"{game.current_player} wins!", grid_size)
                        game_over = True
                    elif game.is_draw():
                        show_message(screen, "It's a draw!", grid_size)
                        game_over = True
                    else:
                        game.current_player = (
                            PLAYER_O if game.current_player == PLAYER_X else PLAYER_X
                        )
            else:
                button_rect = draw_play_again_button(screen, mouse_pos, grid_size)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif (
                        event.type == pygame.MOUSEBUTTONDOWN
                        and button_rect.collidepoint(event.pos)
                    ):
                        return

            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    main()

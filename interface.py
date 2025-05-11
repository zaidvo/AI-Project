import sys
import pygame
from copy import deepcopy

from config import *
from game import TicTacToe
from agent import MinimaxAgent

pygame.init()
pygame.display.set_caption("Enhanced Tic-Tac-Toe")


BACKGROUND = (240, 240, 245)
GRID_COLOR = (180, 180, 200)
HOVER_COLOR = (220, 220, 230)
BUTTON_COLOR = (100, 130, 200)
BUTTON_HOVER = (130, 160, 230)
BUTTON_TEXT = (255, 255, 255)
WIN_COLOR = (100, 200, 100)
X_COLOR = (70, 130, 180)
O_COLOR = (220, 100, 120)
TEXT_COLOR = (60, 60, 80)


title_font = pygame.font.SysFont("Arial", 48, bold=True)
font = pygame.font.SysFont("Arial", 60, bold=True)
medium_font = pygame.font.SysFont("Arial", 36)
small_font = pygame.font.SysFont("Arial", 24)


ANIMATION_SPEED = 10


class Button:
    def __init__(self, x, y, width, height, text, font=medium_font, radius=10):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.radius = min(radius, min(width, height) // 4)
        self.animation = 0
        self.hovered = False
        self.scale_font()

    def scale_font(self):
        """Scale the font to fit within the button width"""
        size = self.font.get_height()

        test_surf = self.font.render(self.text, True, BUTTON_TEXT)

        while test_surf.get_width() > self.rect.width - 20 and size > 16:
            size -= 2
            test_font = pygame.font.SysFont("Arial", size, bold=True)
            test_surf = test_font.render(self.text, True, BUTTON_TEXT)

        if size != self.font.get_height():
            self.font = pygame.font.SysFont("Arial", size, bold=True)

    def draw(self, screen, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

        if self.hovered and self.animation < 5:
            self.animation += 0.5
        elif not self.hovered and self.animation > 0:
            self.animation -= 0.5

        color = [min(255, c + self.animation * 6) for c in BUTTON_COLOR]
        pygame.draw.rect(screen, color, self.rect, border_radius=self.radius)

        pygame.draw.rect(
            screen, GRID_COLOR, self.rect, width=2, border_radius=self.radius
        )

        text_surf = self.font.render(self.text, True, BUTTON_TEXT)
        offset = 1 if self.hovered else 0
        screen.blit(
            text_surf,
            (
                self.rect.centerx - text_surf.get_width() // 2,
                self.rect.centery - text_surf.get_height() // 2 - offset,
            ),
        )

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(
            event.pos
        )


class MenuScreen:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.options = {}
        self.title = ""

    def setup(self, title, options):
        self.title = title
        self.options = {}

        btn_width = min(250, self.screen_width * 0.8)
        btn_height = min(60, self.screen_height * 0.07)
        spacing = min(20, self.screen_height * 0.025)

        total_height = len(options) * (btn_height + spacing) - spacing

        available_height = self.screen_height - 150
        if total_height > available_height:
            btn_height = max(
                40, (available_height - (len(options) - 1) * 10) / len(options)
            )
            spacing = 10
            total_height = len(options) * (btn_height + spacing) - spacing

        y_start = (self.screen_height - total_height) // 2 + 20

        for i, (key, text) in enumerate(options):
            x = (self.screen_width - btn_width) // 2
            y = y_start + i * (btn_height + spacing)
            self.options[key] = Button(x, y, btn_width, btn_height, text)

    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            for key, button in self.options.items():
                if button.is_clicked(event):
                    return key

        return None

    def draw(self, screen):
        screen.fill(BACKGROUND)

        title_surf = title_font.render(self.title, True, TEXT_COLOR)
        shadow_surf = title_font.render(self.title, True, GRID_COLOR)

        title_x = (self.screen_width - title_surf.get_width()) // 2
        title_y = 60

        screen.blit(shadow_surf, (title_x + 2, title_y + 2))
        screen.blit(title_surf, (title_x, title_y))

        line_width = min(500, self.screen_width - 100)
        pygame.draw.line(
            screen,
            GRID_COLOR,
            (
                (self.screen_width - line_width) // 2,
                title_y + title_surf.get_height() + 10,
            ),
            (
                (self.screen_width + line_width) // 2,
                title_y + title_surf.get_height() + 10,
            ),
            3,
        )

        mouse_pos = pygame.mouse.get_pos()
        for button in self.options.values():
            button.draw(screen, mouse_pos)


class GameBoard:
    def __init__(self, game, board_size, cell_size):
        self.game = game
        self.board_size = board_size
        self.cell_size = cell_size
        self.width = board_size * cell_size
        self.height = board_size * cell_size
        self.winning_line = None
        self.animations = {}

    def reset(self, new_game=None):
        if new_game:
            self.game = new_game
        self.winning_line = None
        self.animations = {}

    def draw(self, screen, hover_cell=None):
        pygame.draw.rect(screen, BACKGROUND, (0, 0, self.width, self.height))

        for r in range(1, self.board_size):
            y = r * self.cell_size
            pygame.draw.line(screen, GRID_COLOR, (0, y), (self.width, y), LINE_WIDTH)

            x = r * self.cell_size
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, self.height), LINE_WIDTH)

        if hover_cell:
            r, c = hover_cell
            if (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.game.board[r][c] == EMPTY
            ):
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(screen, HOVER_COLOR, rect)

        for r in range(self.board_size):
            for c in range(self.board_size):
                cell_value = self.game.board[r][c]
                if cell_value != EMPTY:
                    cell_pos = (r, c)

                    if cell_pos not in self.animations:
                        self.animations[cell_pos] = 0

                    if self.animations[cell_pos] < 100:
                        self.animations[cell_pos] += ANIMATION_SPEED

                    color = (
                        WIN_COLOR
                        if self.winning_line and cell_pos in self.winning_line
                        else (X_COLOR if cell_value == PLAYER_X else O_COLOR)
                    )

                    center_x = c * self.cell_size + self.cell_size // 2
                    center_y = r * self.cell_size + self.cell_size // 2
                    progress = min(100, self.animations[cell_pos]) / 100

                    if cell_value == PLAYER_X:
                        self._draw_x(screen, center_x, center_y, progress, color)
                    else:
                        self._draw_o(screen, center_x, center_y, progress, color)

    def _draw_x(self, screen, center_x, center_y, progress, color):
        size = int(self.cell_size * 0.6 * progress)
        thickness = max(5, int(self.cell_size * 0.08))
        offset = size // 2

        pygame.draw.line(
            screen,
            color,
            (center_x - offset, center_y - offset),
            (center_x + offset, center_y + offset),
            thickness,
        )
        pygame.draw.line(
            screen,
            color,
            (center_x + offset, center_y - offset),
            (center_x - offset, center_y + offset),
            thickness,
        )

        cap_radius = thickness // 2
        points = [
            (center_x - offset, center_y - offset),
            (center_x + offset, center_y + offset),
            (center_x + offset, center_y - offset),
            (center_x - offset, center_y + offset),
        ]
        for x, y in points:
            pygame.draw.circle(screen, color, (x, y), cap_radius)

    def _draw_o(self, screen, center_x, center_y, progress, color):
        radius = int(self.cell_size * 0.3 * progress)
        thickness = max(5, int(self.cell_size * 0.08))

        pygame.draw.circle(screen, color, (center_x, center_y), radius, thickness)

    def set_winning_line(self, winning_line):
        self.winning_line = winning_line

    def get_cell_at_pos(self, pos):
        x, y = pos
        r, c = y // self.cell_size, x // self.cell_size
        if 0 <= r < self.board_size and 0 <= c < self.board_size:
            return (r, c)
        return None


class GameUI:
    def __init__(self, screen_width, screen_height, grid_size=5):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_size = grid_size
        self.cell_size = min(
            screen_width // grid_size, (screen_height - 120) // grid_size
        )

        board_width = grid_size * self.cell_size
        board_height = grid_size * self.cell_size
        self.board_x = (screen_width - board_width) // 2
        self.board_y = 100

        self.game = TicTacToe(grid_size)
        self.board = GameBoard(self.game, grid_size, self.cell_size)

        self.game_over = False
        self.winner = None
        self.menu = MenuScreen(screen_width, screen_height)

        btn_width = min(180, screen_width * 0.25)
        btn_height = min(50, screen_height * 0.06)
        btn_spacing = min(10, screen_height * 0.015)

        self.play_again_btn = Button(
            (screen_width - btn_width) // 2,
            self.board_y + board_height + 20,
            btn_width,
            btn_height,
            "Play Again",
        )

        self.menu_btn = Button(
            (screen_width - btn_width) // 2,
            self.board_y + board_height + 20 + btn_height + btn_spacing,
            btn_width,
            btn_height,
            "Main Menu",
            small_font,
        )

        self.quit_btn = Button(
            (screen_width - btn_width) // 2,
            self.board_y + board_height + 20 + (btn_height + btn_spacing) * 2,
            btn_width,
            btn_height,
            "Quit Game",
            small_font,
        )

        self.status_message = ""
        self.status_alpha = 0

    def reset_game(self):
        self.game = TicTacToe(self.grid_size)
        self.board.reset(self.game)
        self.game_over = False
        self.winner = None
        self.update_status(f"{self.game.current_player}'s Turn")

    def update_status(self, message):
        self.status_message = message
        self.status_alpha = 0

    def draw_status(self, screen):
        if self.status_alpha < 255:
            self.status_alpha += 10

        status_font = medium_font
        status_surf = status_font.render(self.status_message, True, TEXT_COLOR)
        status_surf.set_alpha(self.status_alpha)

        x = (self.screen_width - status_surf.get_width()) // 2
        y = 60
        screen.blit(status_surf, (x, y))

    def handle_human_move(self, pos):
        adjusted_pos = (pos[0] - self.board_x, pos[1] - self.board_y)
        cell = self.board.get_cell_at_pos(adjusted_pos)

        if cell and not self.game_over:
            return self.game.make_move(cell, self.game.current_player)
        return False

    def handle_ai_move(self, agent):
        self.update_status("AI is thinking...")
        pygame.display.flip()
        pygame.time.delay(300)

        move = agent.choose_move(deepcopy(self.game))
        if move:
            return self.game.make_move(move, self.game.current_player)
        return False

    def check_game_end(self):
        current_player = self.game.current_player

        if self.game.check_winner(current_player):
            winning_line = get_winning_line(self.game, self.grid_size, current_player)
            self.board.set_winning_line(winning_line)
            self.update_status(f"{current_player} Wins!")
            self.game_over = True
            self.winner = current_player
            return True

        elif self.game.is_draw():
            self.update_status("It's a Draw!")
            self.game_over = True
            return True

        return False

    def draw_game_screen(self, screen, hover_pos=None):
        screen.fill(BACKGROUND)

        title_text = "Tic-Tac-Toe"
        title_surf = title_font.render(title_text, True, TEXT_COLOR)
        screen.blit(title_surf, ((self.screen_width - title_surf.get_width()) // 2, 10))

        self.draw_status(screen)

        hover_cell = None
        if hover_pos and not self.game_over:
            mouse_x, mouse_y = hover_pos
            adjusted_pos = (mouse_x - self.board_x, mouse_y - self.board_y)
            hover_cell = self.board.get_cell_at_pos(adjusted_pos)
            if hover_cell and self.game.board[hover_cell[0]][hover_cell[1]] != EMPTY:
                hover_cell = None

        pygame.draw.rect(
            screen,
            (230, 230, 240),
            (
                self.board_x - 10,
                self.board_y - 10,
                self.board.width + 20,
                self.board.height + 20,
            ),
            border_radius=5,
        )

        pygame.draw.rect(
            screen,
            GRID_COLOR,
            (
                self.board_x - 10,
                self.board_y - 10,
                self.board.width + 20,
                self.board.height + 20,
            ),
            width=2,
            border_radius=5,
        )

        orig_clip = screen.get_clip()
        screen.set_clip(
            pygame.Rect(self.board_x, self.board_y, self.board.width, self.board.height)
        )

        temp_surf = screen.subsurface(
            self.board_x, self.board_y, self.board.width, self.board.height
        )
        self.board.draw(temp_surf, hover_cell)
        screen.set_clip(orig_clip)

        if self.game_over:
            mouse_pos = pygame.mouse.get_pos()
            self.play_again_btn.draw(screen, mouse_pos)
            self.menu_btn.draw(screen, mouse_pos)
            self.quit_btn.draw(screen, mouse_pos)


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


def main():
    base_width, base_height = 700, 800
    screen = pygame.display.set_mode((base_width, base_height))

    ui = None
    game_state = "main_menu"
    clock = pygame.time.Clock()
    menu = MenuScreen(base_width, base_height)

    ai_depth = 3
    grid_size = 5
    player_first = True
    game_mode = "Human vs Human"

    while True:
        if game_state == "main_menu":
            menu.setup(
                "Tic-Tac-Toe",
                [
                    ("play", "Play Game"),
                    ("settings", "Settings"),
                    ("quit", "Quit"),
                ],
            )

            choice = menu.handle_events()
            menu.draw(screen)

            if choice == "play":
                game_state = "mode_select"
            elif choice == "settings":
                game_state = "settings"
            elif choice == "quit":
                pygame.quit()
                sys.exit()

        elif game_state == "mode_select":
            menu.setup(
                "Select Game Mode",
                [
                    ("hvh", "Human vs Human"),
                    ("hvc", "Human vs AI"),
                    ("cvc", "AI vs AI"),
                    ("back", "Back"),
                ],
            )

            choice = menu.handle_events()
            menu.draw(screen)

            if choice == "hvh":
                game_mode = "Human vs Human"
                game_state = "start_game"
            elif choice == "hvc":
                game_mode = "Human vs AI"
                game_state = "turn_select"
            elif choice == "cvc":
                game_mode = "AI vs AI"
                game_state = "depth_select"
            elif choice == "back":
                game_state = "main_menu"

        elif game_state == "turn_select":
            menu.setup(
                "Who Goes First?",
                [("player", "Player First"), ("ai", "AI First"), ("back", "Back")],
            )

            choice = menu.handle_events()
            menu.draw(screen)

            if choice == "player":
                player_first = True
                game_state = "depth_select"
            elif choice == "ai":
                player_first = False
                game_state = "depth_select"
            elif choice == "back":
                game_state = "mode_select"

        elif game_state == "depth_select":
            menu.setup(
                "Select AI Difficulty",
                [
                    ("easy", "Easy"),
                    ("medium", "Medium"),
                    ("difficult", "Difficult"),
                    ("impossible", "Impossible"),
                    ("back", "Back"),
                ],
            )

            choice = menu.handle_events()
            menu.draw(screen)

            if choice == "easy":
                ai_depth = 3
                game_state = "start_game"
            elif choice == "medium":
                ai_depth = 4
                game_state = "start_game"
            elif choice == "difficult":
                ai_depth = 5
                game_state = "start_game"
            elif choice == "impossible":
                ai_depth = 6
                game_state = "start_game"
            elif choice == "back":
                game_state = (
                    "turn_select" if game_mode == "Human vs AI" else "mode_select"
                )

        elif game_state == "settings":
            menu.setup("Settings", [("back", "Back to Main Menu")])

            choice = menu.handle_events()
            menu.draw(screen)

            if choice == "back":
                game_state = "main_menu"

        elif game_state == "start_game":
            cell_size = 120
            screen_width = max(700, grid_size * cell_size + 100)
            screen_height = max(800, grid_size * cell_size + 240)

            screen = pygame.display.set_mode((screen_width, screen_height))

            ui = GameUI(screen_width, screen_height, grid_size)
            ui.reset_game()

            agent_X = (
                MinimaxAgent(PLAYER_X, PLAYER_O, ai_depth)
                if "AI" in game_mode and (game_mode == "AI vs AI" or not player_first)
                else None
            )
            agent_O = (
                MinimaxAgent(PLAYER_O, PLAYER_X, ai_depth)
                if "AI" in game_mode and (game_mode == "AI vs AI" or player_first)
                else None
            )

            ui.game.current_player = PLAYER_X

            game_state = "playing"

        elif game_state == "playing":
            mouse_pos = pygame.mouse.get_pos()
            current_agent = agent_X if ui.game.current_player == PLAYER_X else agent_O

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if ui.game_over:
                    if ui.play_again_btn.is_clicked(event):
                        ui.reset_game()
                    elif ui.menu_btn.is_clicked(event):
                        game_state = "main_menu"
                        screen = pygame.display.set_mode((base_width, base_height))
                    elif ui.quit_btn.is_clicked(event):
                        pygame.quit()
                        sys.exit()

                elif not current_agent and event.type == pygame.MOUSEBUTTONDOWN:
                    if ui.handle_human_move(event.pos):
                        if not ui.check_game_end():
                            ui.game.current_player = (
                                PLAYER_O
                                if ui.game.current_player == PLAYER_X
                                else PLAYER_X
                            )
                            ui.update_status(f"{ui.game.current_player}'s Turn")

            if not ui.game_over and current_agent and game_state == "playing":
                if ui.handle_ai_move(current_agent):
                    if not ui.check_game_end():
                        ui.game.current_player = (
                            PLAYER_O if ui.game.current_player == PLAYER_X else PLAYER_X
                        )
                        ui.update_status(f"{ui.game.current_player}'s Turn")

            ui.draw_game_screen(screen, mouse_pos)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()

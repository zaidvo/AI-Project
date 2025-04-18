# game/constants.py

# Board constants
BOARD_SIZE = 7
WIN_LENGTH = 5

# Player constants
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

# Game states
ONGOING = 0
X_WINS = 1
O_WINS = 2
DRAW = 3

# AI constants
REWARD_WIN = 10.0
REWARD_LOSS = -10.0
REWARD_DRAW = 0.0
REWARD_INVALID_MOVE = -100.0

# Display constants
CELL_SIZE = 80
BOARD_PADDING = 20
WIDTH = BOARD_SIZE * CELL_SIZE + 2 * BOARD_PADDING
HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * BOARD_PADDING + 100  # Extra space for UI
BG_COLOR = (240, 240, 240)
GRID_COLOR = (80, 80, 80)
HIGHLIGHT_COLOR = (255, 255, 0, 128)

# X and O colors
X_COLOR = (41, 128, 185)  # Blue
O_COLOR = (231, 76, 60)   # Red

# UI constants
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
BUTTON_COLOR = (52, 152, 219)
BUTTON_HOVER_COLOR = (41, 128, 185)
BUTTON_TEXT_COLOR = (255, 255, 255)
FONT_SIZE = 24
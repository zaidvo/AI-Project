# game/__init__.py
"""
Game package for MegaTicTacToe.
Contains board implementation and game mechanics.
"""

from .board import Board
from .constants import *

__all__ = ['Board', 'EMPTY', 'PLAYER_X', 'PLAYER_O', 'BOARD_SIZE', 'WIN_LENGTH']
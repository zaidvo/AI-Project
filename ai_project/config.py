# models/config.py
"""
Configuration for DQN model used in MegaTicTacToe.
"""

# Board configuration
BOARD_SIZE = 7
WIN_LENGTH = 5

# DQN configuration
DQN_CONFIG = {
    'state_size': BOARD_SIZE * BOARD_SIZE,
    'action_size': BOARD_SIZE * BOARD_SIZE,
    'hidden_layers': [128, 64],
    'learning_rate': 0.001,
    'gamma': 0.95,            # Discount factor
    'epsilon': 1.0,           # Starting exploration rate
    'epsilon_decay': 0.995,   # Decay rate for exploration
    'epsilon_min': 0.01,      # Minimum exploration rate
    'memory_size': 10000,     # Size of replay buffer
    'batch_size': 64,         # Training batch size
    'target_update_freq': 1000  # Target network update frequency
}

# Training configuration
TRAINING_CONFIG = {
    'episodes': 20000,         # Total training episodes
    'save_interval': 1000,     # How often to save model
    'print_interval': 100,     # How often to print progress
    'plot_metrics': True       # Whether to plot training metrics
}

# Reward configuration
REWARD_CONFIG = {
    'win': 10.0,
    'loss': -10.0,
    'draw': 0.0,
    'invalid_move': -100.0
}

# Model file paths
MODEL_PATH = 'dqn_megatictactoe.h5'
METRICS_PATH = 'training_metrics.pkl'
PLOTS_DIR = 'plots'
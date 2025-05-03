# AlphaZero for 5x5 TicTacToe (4-in-a-row)

This project implements the AlphaZero algorithm for a 5x5 TicTacToe game where connecting 4 stones in a row (horizontally, vertically, or diagonally) wins the game.

## Features

- Optimized for 5x5 board with 4-in-a-row win condition
- Monte Carlo Tree Search (MCTS) with neural network guidance
- Self-play training pipeline
- Deep residual neural network architecture
- Play against trained agent
- Tournament system to compare different checkpoints

## Project Structure

- `env.py`: TicTacToe environment with customizable board size and win condition
- `model.py`: Neural network architecture for policy and value prediction
- `mcts.py`: Monte Carlo Tree Search implementation
- `agent.py`: AlphaZero agent with self-play and training capabilities
- `train.py`: Training script to run the AlphaZero training loop
- `play.py`: Play against the trained agent
- `tournament.py`: Run tournaments between different model checkpoints

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- tqdm

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/alphazero-tictactoe.git
cd alphazero-tictactoe

# Install dependencies
pip install torch numpy tqdm
```

## Training

To train the agent from scratch:

```bash
python train.py --board_size 5 --win_length 4 --iterations 50 --games 100
```

Parameters:

- `--board_size`: Size of the board (default: 5)
- `--win_length`: Number of stones in a row needed to win (default: 4)
- `--iterations`: Number of training iterations (default: 50)
- `--games`: Self-play games per iteration (default: 50)
- `--simulations`: MCTS simulations per move (default: 800)
- `--epochs`: Training epochs per iteration (default: 10)
- `--checkpoint_interval`: Save checkpoint every N iterations (default: 1)
- `--no_resume`: Don't resume from existing checkpoint
- `--evaluate_every`: Evaluate against random player every N iterations (default: 5)
- `--eval_games`: Number of evaluation games (default: 20)

## Playing Against the Agent

To play against a trained agent:

```bash
python play.py --checkpoint checkpoints_5x5_4in/alphazero_tictactoe_5x5_4in_best.pt --player 1
```

Parameters:

- `--board_size`: Size of the board (default: 5)
- `--win_length`: Number of stones in a row needed to win (default: 4)
- `--checkpoint_dir`: Directory with checkpoints (default: ./checkpoints)
- `--checkpoint`: Specific checkpoint to load (default: most recent)
- `--player`: Human player: 1=X (first), -1=O (second) (default: 1)
- `--simulations`: Number of MCTS simulations per move (default: 800)

## Running Tournaments

To compare different checkpoints in a tournament:

```bash
python tournament.py --checkpoints_dir checkpoints_5x5_4in --games 10
```

Parameters:

- `--checkpoints_dir`: Directory containing checkpoints (required)
- `--board_size`: Size of the board (default: 5)
- `--win_length`: Win condition length (default: 4)
- `--simulations`: MCTS simulations per move (default: 800)
- `--games`: Games per match (default: 10)

## Optimizations

Several optimizations have been implemented for the 5x5 game with 4-in-a-row win condition:

- Tuned neural network depth and width
- Improved MCTS exploration parameters
- Dirichlet noise for exploration diversity
- Learning rate scheduling
- Efficient win checking algorithm
- KL divergence loss for policy training

## Performance

The trained agent can:

- Learn effective strategies for 5x5 TicTacToe
- Block opponent's threats
- Create strategic patterns to form winning moves
- Evaluate board positions accurately

## Acknowledgements

This implementation is inspired by the AlphaZero algorithm described in the paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" by Silver et al.

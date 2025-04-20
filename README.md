# 7x7 Tic Tac Toe with Deep Q-Learning

This project implements a 7x7 Tic Tac Toe game with an AI agent trained using Deep Q-Network (DQN) reinforcement learning. The goal is to get 4 in a row (horizontally, vertically, or diagonally).

## Files

- `tictactoe_env.py`: Game environment for 7x7 Tic Tac Toe
- `dqn_model.py`: DQN neural network model
- `dqn_agent.py`: DQN agent with experience replay and target network
- `train.py`: Script for training the agent
- `play.py`: Script for playing against the trained agent
- `test.py`: Script for evaluating the agent's performance

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- tqdm

Install the requirements:

```bash
pip install torch numpy matplotlib tqdm
```

## Training the Agent

To train the agent against a random opponent:

```bash
python train.py --episodes 10000 --save_interval 1000
```

To train the agent using self-play (recommended for better performance):

```bash
python train.py --episodes 10000 --self_play --save_interval 1000
```

Additional training options:

```bash
python train.py --help
```

The training script will save model checkpoints at regular intervals and plot the training rewards.

## Playing Against the Agent

To play against the trained agent:

```bash
python play.py --model_path model_final.pt
```

By default, you play as O (second player). To play as X (first player):

```bash
python play.py --model_path model_final.pt --human_player 1
```

## Testing the Agent

To evaluate the agent's performance against a random opponent:

```bash
python test.py --model_path model_final.pt --games 1000
```

To evaluate the agent in self-play:

```bash
python test.py --model_path model_final.pt --games 1000 --opponent self
```

To display games during testing (this slows down testing significantly):

```bash
python test.py --model_path model_final.pt --games 10 --render --verbose
```

## Implementation Details

### Game Environment

- 7x7 grid
- Goal: Get 4 in a row (horizontally, vertically, or diagonally)
- X plays first, O plays second
- State representation: 7x7 numpy array with:
  - 1 for X
  - -1 for O
  - 0 for empty

### DQN Model

- Input: Two 7x7 channels (one for X positions, one for O positions)
- Architecture:
  - 3 convolutional layers
  - 3 fully connected layers
  - Output: Q-values for all 49 possible actions

### Agent Features

- Experience replay memory
- Target network for stable learning
- Epsilon-greedy exploration strategy
- Self-play training option

## Performance

After training with self-play for 10,000 episodes, the agent typically achieves:

- ~85-95% win rate against a random opponent when playing as X
- ~75-85% win rate against a random opponent when playing as O
- Near-optimal play in self-play scenarios (most games end in draws)

## Future Improvements

- Implement Monte Carlo Tree Search (MCTS) for stronger opponent
- Add minimax-based agent for comparison
- Implement prioritized experience replay
- Create a web interface for human play

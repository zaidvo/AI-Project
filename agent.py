import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from tqdm import tqdm
import random

from model import AlphaZeroNetwork
from mcts import MCTS
from env import TicTacToeEnv


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class AlphaZeroAgent:
    def __init__(
        self,
        board_size=5,
        win_length=4,
        num_simulations=800,
        batch_size=128,
        lr=0.001,
        weight_decay=1e-4,
        checkpoint_dir="./checkpoints",
        device=None,
    ):
        # Game parameters
        self.board_size = board_size
        self.win_length = win_length
        self.num_simulations = num_simulations

        # Training parameters
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir

        # Set device (CPU/GPU)
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        # Initialize network
        self.network = AlphaZeroNetwork(board_size=board_size).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.1
        )

        # Initialize MCTS
        self.mcts = MCTS(
            self.network,
            self.device,
            num_simulations=num_simulations,
            board_size=board_size,
            win_length=win_length,
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def select_action(self, env, temperature=0.0, add_exploration_noise=False):
        """Select action using MCTS search"""
        root = self.mcts.search(env, add_exploration_noise=add_exploration_noise)
        action = root.select_action(temperature)
        policy = root.get_visit_count_policy()

        return action, policy

    def self_play(self, num_games=100, temperature_threshold=10):
        """Perform self-play games to generate training data"""
        game_pbar = tqdm(range(num_games), desc="Self-play games")

        total_steps = 0
        outcomes = {1: 0, -1: 0, 0: 0}  # Track X wins, O wins, draws

        for game in game_pbar:
            env = TicTacToeEnv(board_size=self.board_size, win_length=self.win_length)
            game_history = []
            step = 0

            # Play until game is done
            while not env.done:
                # Determine temperature - use higher value early in the game
                temperature = 1.0 if step < temperature_threshold else 0.0

                # Add exploration noise at root for additional randomness
                add_noise = step == 0

                # Select action using MCTS
                action, policy = self.select_action(
                    env, temperature=temperature, add_exploration_noise=add_noise
                )

                # Store state and policy before making the move
                observation = env.get_observation()
                game_history.append((observation, policy, env.current_player))

                # Make the move
                _, reward, done, info = env.step(action)
                step += 1

            # Game finished - determine outcome
            outcome = info["winner"]
            outcomes[outcome] += 1
            total_steps += step

            # Update progress bar with stats
            game_pbar.set_postfix(
                {
                    "steps": step,
                    "winner": "X" if outcome == 1 else "O" if outcome == -1 else "Draw",
                    "avg_len": round(total_steps / (game + 1), 1),
                }
            )

            # Add experiences to replay buffer with appropriate value targets
            for i, (observation, pi, player) in enumerate(game_history):
                # Set value target based on game outcome from player's perspective
                if outcome == 0:  # Draw
                    value = 0
                else:  # Win/loss
                    value = 1 if outcome == player else -1

                self.replay_buffer.add((observation, pi, value))

        # Print self-play summary
        print(f"\nSelf-play summary:")
        print(f"Total games: {num_games}")
        print(f"X wins: {outcomes[1]} ({outcomes[1]/num_games*100:.1f}%)")
        print(f"O wins: {outcomes[-1]} ({outcomes[-1]/num_games*100:.1f}%)")
        print(f"Draws: {outcomes[0]} ({outcomes[0]/num_games*100:.1f}%)")
        print(f"Average game length: {total_steps/num_games:.1f} steps")

        return outcomes

    def train(self, epochs=10, verbose=True):
        """Train network on data in replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            print(
                f"Not enough data in replay buffer to train. Have {len(self.replay_buffer)}, need {self.batch_size}."
            )
            return

        self.network.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        progress_bar = (
            tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        )

        for epoch in progress_bar:
            # Sample batch from replay buffer
            batch = self.replay_buffer.sample(self.batch_size)
            observations, pis, values = zip(*batch)

            # Convert to tensors
            observations = torch.FloatTensor(np.array(observations)).to(self.device)
            pis = torch.FloatTensor(np.array(pis)).to(self.device)
            values = torch.FloatTensor(np.array(values).reshape(-1, 1)).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            policy_logits, value_pred = self.network(observations)

            # Calculate loss
            # Use KL divergence for policy loss (better for matching distributions)
            policy_loss = -torch.mean(
                torch.sum(pis * F.log_softmax(policy_logits, dim=1), dim=1)
            )
            value_loss = F.mse_loss(value_pred, values)
            loss = policy_loss + value_loss

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

            if verbose:
                progress_bar.set_postfix(
                    {
                        "loss": loss.item(),
                        "p_loss": policy_loss.item(),
                        "v_loss": value_loss.item(),
                    }
                )

        # Update learning rate
        self.scheduler.step()

        # Print training summary
        if verbose:
            print(f"Training summary:")
            print(f"Avg loss: {total_loss/epochs:.4f}")
            print(f"Avg policy loss: {total_policy_loss/epochs:.4f}")
            print(f"Avg value loss: {total_value_loss/epochs:.4f}")
            print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

        self.network.eval()
        return total_loss / epochs

    def save_checkpoint(self, filename=None):
        """Save model checkpoint"""
        if filename is None:
            filename = f"alphazero_tictactoe_{self.board_size}x{self.board_size}_{int(time.time())}.pt"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "board_size": self.board_size,
                "win_length": self.win_length,
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Check compatibility
        if (
            checkpoint["board_size"] != self.board_size
            or checkpoint.get("win_length", self.win_length) != self.win_length
        ):
            print(
                f"Warning: Checkpoint parameters ({checkpoint['board_size']}x{checkpoint.get('win_length', 3)}) "
                + f"don't match current parameters ({self.board_size}x{self.win_length})"
            )

        # Load model state
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler if available
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def evaluate_against_random(self, num_games=100, render=False):
        """Evaluate agent against a random player"""
        wins = 0
        losses = 0
        draws = 0

        for game in tqdm(range(num_games), desc="Evaluation games"):
            env = TicTacToeEnv(board_size=self.board_size, win_length=self.win_length)

            # Randomly decide who goes first
            player_agent = random.choice([1, -1])  # 1=X, -1=O

            while not env.done:
                if render:
                    env.render()

                if env.current_player == player_agent:
                    # Agent's turn
                    action, _ = self.select_action(env, temperature=0)
                    _, _, _, _ = env.step(action)
                else:
                    # Random player's turn
                    valid_moves = np.where(env.get_valid_moves_flat() == 1)[0]
                    random_action = np.random.choice(valid_moves)
                    _, _, _, _ = env.step(random_action)

            # Game finished
            if env.winner == player_agent:
                wins += 1
            elif env.winner == 0:
                draws += 1
            else:
                losses += 1

            if render:
                env.render()
                print(
                    f"Game {game+1} result: {'Win' if env.winner == player_agent else 'Draw' if env.winner == 0 else 'Loss'}"
                )

        win_rate = wins / num_games * 100
        print(f"\nEvaluation results against random player:")
        print(f"Wins: {wins} ({win_rate:.1f}%)")
        print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
        print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")

        return win_rate

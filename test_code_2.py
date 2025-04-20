import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Tic-Tac-Toe 7x7 Environment
class TicTacToe7x7:
    def __init__(self):
        self.board = np.zeros((7, 7), dtype=int)  # 0: empty, 1: X, -1: O
        self.current_player = 1  # X starts

    def reset(self):
        self.board = np.zeros((7, 7), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        # Return state as 3 channels: X, O, empty
        x_channel = (self.board == 1).astype(float)
        o_channel = (self.board == -1).astype(float)
        empty_channel = (self.board == 0).astype(float)
        return np.stack([x_channel, o_channel, empty_channel], axis=0)

    def get_valid_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, move):
        if self.board[move] == 0:
            self.board[move] = self.current_player
            self.current_player = -self.current_player
            return True
        return False

    def check_winner(self):
        # Check rows, columns, and diagonals for 5-in-a-row (adjustable)
        for i in range(7):
            for j in range(3):  # For 5-in-a-row in a 7x7 grid
                # Rows
                if np.all(self.board[i, j : j + 5] == 1) or np.all(
                    self.board[i, j : j + 5] == -1
                ):
                    return self.board[i, j]
                # Columns
                if np.all(self.board[j : j + 5, i] == 1) or np.all(
                    self.board[j : j + 5, i] == -1
                ):
                    return self.board[j, i]
        # Main diagonals
        for i in range(3):
            for j in range(3):
                if np.all([self.board[i + k, j + k] == 1 for k in range(5)]) or np.all(
                    [self.board[i + k, j + k] == -1 for k in range(5)]
                ):
                    return self.board[i, j]
        # Anti-diagonals
        for i in range(3):
            for j in range(4, 7):
                if np.all([self.board[i + k, j - k] == 1 for k in range(5)]) or np.all(
                    [self.board[i + k, j - k] == -1 for k in range(5)]
                ):
                    return self.board[i, j]
        return 0

    def is_done(self):
        return len(self.get_valid_moves()) == 0 or self.check_winner() != 0


# CNN for DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 49)  # 7x7 = 49 possible actions

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.array(state), action, reward, np.array(next_state), done

    def _len_(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 1000

    def select_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            valid_q_values = q_values[0, [i * 7 + j for i, j in valid_moves]]
            action_idx = valid_q_values.argmax().item()
            return valid_moves[action_idx]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor([i * 7 + j for i, j in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, episode, filename="checkpoint.pth"):
        checkpoint = {
            "episode": episode,
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer.buffer),
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self, filename="checkpoint.pth"):
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.replay_buffer.buffer = deque(checkpoint["replay_buffer"], maxlen=10000)
            self.epsilon = checkpoint["epsilon"]
            print(f"Loaded checkpoint from episode {checkpoint['episode']}")
            return checkpoint["episode"]
        return 0


# Training Loop
def train_dqn(episodes=10000, checkpoint_interval=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToe7x7()
    agent = DQNAgent(device)
    start_episode = agent.load_checkpoint()  # Resume from checkpoint if exists
    step_count = 0
    stats = {"wins": 0, "draws": 0, "losses": 0}

    for episode in range(start_episode, episodes):
        state = env.reset()
        done = False
        while not done:
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                break

            # Player X (DQN)
            action = agent.select_action(state, valid_moves)
            env.make_move(action)
            next_state = env.get_state()
            winner = env.check_winner()
            done = env.is_done()

            if winner == 1:
                reward = 1
                stats["wins"] += 1
            elif winner == -1:
                reward = -1
                stats["losses"] += 1
            elif done:
                reward = 0
                stats["draws"] += 1
            else:
                reward = 0

            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            step_count += 1

            if step_count % agent.target_update_freq == 0:
                agent.update_target()

            if done:
                break

            # Player O (Random or another DQN for self-play)
            valid_moves = env.get_valid_moves()
            if valid_moves:
                action = random.choice(valid_moves)  # Random opponent for simplicity
                env.make_move(action)
                state = env.get_state()
                winner = env.check_winner()
                done = env.is_done()

                if winner == -1:
                    agent.replay_buffer.push(state, action, -1, state, True)
                    stats["losses"] += 1
                elif done:
                    agent.replay_buffer.push(state, action, 0, state, True)
                    stats["draws"] += 1

        agent.decay_epsilon()

        # Print stats every 100 episodes
        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}, Epsilon: {agent.epsilon:.3f}, "
                f"Wins: {stats['wins']}, Draws: {stats['draws']}, Losses: {stats['losses']}"
            )
            stats = {"wins": 0, "draws": 0, "losses": 0}

        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            agent.save_checkpoint(episode + 1)


if __name__ == "__main__":
    train_dqn(episodes=10000, checkpoint_interval=1000)

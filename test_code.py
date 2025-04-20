import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


# Environment
class TicTacToe7x7:
    def __init__(self):
        self.board = np.zeros((7, 7), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.winner = None
        self.done = False

    def reset(self):
        self.board = np.zeros((7, 7), dtype=int)
        self.current_player = 1
        self.winner = None
        self.done = False
        return self.get_state()

    def get_state(self):
        # Returns a 3x7x7 one-hot encoded state (channels: X, O, empty)
        state = np.zeros((3, 7, 7), dtype=int)
        state[0] = self.board == 1  # X positions
        state[1] = self.board == -1  # O positions
        state[2] = self.board == 0  # Empty positions
        return state

    def step(self, action):
        row, col = action // 7, action % 7
        if self.board[row, col] != 0:
            return self.get_state(), -10, True  # Invalid move penalty

        self.board[row, col] = self.current_player
        reward = 0

        # Check for win (5 in a row, column, or diagonal)
        if self.check_win():
            self.winner = self.current_player
            reward = 1 if self.current_player == 1 else -1
            self.done = True
        elif np.all(self.board != 0):  # Draw
            self.done = True

        self.current_player *= -1  # Switch player
        return self.get_state(), reward, self.done

    def check_win(self):
        # Check rows, columns, and diagonals for 5 consecutive marks
        for i in range(7):
            for j in range(3):
                # Check rows
                if abs(sum(self.board[i, j : j + 5])) == 5:
                    return True
                # Check columns
                if abs(sum(self.board[j : j + 5, i])) == 5:
                    return True
        # Check diagonals
        for i in range(3):
            for j in range(3):
                if abs(sum(self.board[i + k, j + k] for k in range(5))) == 5:
                    return True
                if abs(sum(self.board[i + k, j + 4 - k] for k in range(5))) == 5:
                    return True
        return False


# DQN Model (CNN-based)
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 49)  # 7x7=49 possible actions

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Double DQN Agent
class DQNAgent:
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 1000
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 48)  # Random action (0-48)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, episode, path="checkpoint.pth"):
        torch.save(
            {
                "episode": episode,
                "policy_state": self.policy_net.state_dict(),
                "target_state": self.target_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "memory": self.memory,
            },
            path,
        )
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self, path="checkpoint.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint["policy_state"])
            self.target_net.load_state_dict(checkpoint["target_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.epsilon = checkpoint["epsilon"]
            self.memory = checkpoint["memory"]
            print(f"Resumed training from episode {checkpoint['episode']}")
            return checkpoint["episode"]
        return 0


# Training Loop
def train(episodes=10000, resume=False):
    env = TicTacToe7x7()
    agent = DQNAgent()
    start_episode = 0

    if resume:
        start_episode = agent.load_checkpoint()

    for episode in range(start_episode, episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

        if episode % 100 == 0:
            agent.save_checkpoint(episode)
            print(
                f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}"
            )


if __name__ == "__main__":
    train(episodes=10000, resume=True)

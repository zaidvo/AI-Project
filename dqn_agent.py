import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

from dqn_model import DQN

# Define a transition for memory replay
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory:
    """Memory buffer for experience replay"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    Agent implementing Deep Q-Network with experience replay and target network
    """

    def __init__(
        self,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=10000,
        memory_size=10000,
        batch_size=64,
        target_update=1000,
    ):
        self.board_size = 7  # Define board_size for 7x7 Tic Tac Toe

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main network and target network
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayMemory(memory_size)

        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon_start = epsilon_start  # Initial exploration rate
        self.epsilon_end = epsilon_end  # Final exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.batch_size = batch_size
        self.target_update = target_update  # How often to update target network

        # Step counter
        self.steps_done = 0

    def select_action(self, state, valid_moves):
        """
        Select an action using epsilon-greedy policy

        Args:
            state: Current game state (7x7 numpy array)
            valid_moves: List of valid move indices

        Returns:
            action: Selected action index
        """
        # Calculate current epsilon
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

        self.steps_done += 1

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Random action from valid moves
            return random.choice(valid_moves)
        else:
            # Convert state to model input
            state_tensor = self.policy_net.state_to_input(state).to(self.device)

            with torch.no_grad():
                # Get Q-values for all actions
                q_values = self.policy_net(state_tensor)

                # Set invalid moves to very negative values
                all_moves = list(range(49))  # All possible moves
                invalid_moves = list(set(all_moves) - set(valid_moves))

                if invalid_moves:
                    q_values[0, invalid_moves] = float("-inf")

                # Select best valid action
                action = q_values.max(1)[1].item()

                return action

    def optimize_model(self):
        """Perform one step of optimization on the Q-network"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from replay memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to torch tensors
        state_batch = torch.cat(
            [self.policy_net.state_to_input(s) for s in batch.state]
        ).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float32
        )

        # Create masks for non-terminal states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        # Create tensor for non-terminal next states
        non_final_next_states = torch.cat(
            [
                self.policy_net.state_to_input(s)
                for s in batch.next_state
                if s is not None
            ]
        ).to(self.device)

        # Compute Q-values for the current states and actions
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute expected Q-values
        next_q_values = torch.zeros(self.batch_size, device=self.device)

        if non_final_mask.sum() > 0:
            with torch.no_grad():
                next_q_values[non_final_mask] = self.target_net(
                    non_final_next_states
                ).max(1)[0]

        # Compute target Q-values
        expected_q_values = reward_batch + (
            self.gamma * next_q_values * (~torch.tensor(batch.done, device=self.device))
        )

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update the target network with the parameters from the policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action_target(self, state, valid_moves):
        """Select action using the target network (for self-play)"""
        # Similar to select_action but uses target network
        state_tensor = self.policy_net.state_to_input(state).to(self.device)

        with torch.no_grad():
            q_values = self.target_net(state_tensor).detach().cpu().numpy()[0]

            # Set invalid moves to very negative values
            for i in range(self.board_size * self.board_size):
                if i not in valid_moves:
                    q_values[i] = -1e9

            return np.argmax(q_values)

    def select_action_with_model(self, state, valid_moves, model):
        """Select action using a provided model (for playing against previous versions)"""
        state_tensor = self.policy_net.state_to_input(state).to(self.device)

        with torch.no_grad():
            q_values = model(state_tensor).detach().cpu().numpy()[0]

            # Set invalid moves to very negative values
            for i in range(self.board_size * self.board_size):
                if i not in valid_moves:
                    q_values[i] = -1e9

            return np.argmax(q_values)

    def get_model_copy(self):
        """Get a copy of the current policy network"""
        import copy

        return copy.deepcopy(self.policy_net)

    def save_model(self, path):
        """Save the model to a file"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load_model(self, path):
        """Load the model from a file"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]

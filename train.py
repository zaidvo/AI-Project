import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy

# Import the enhanced model
from dqn_model import EnhancedDQN

# Experience replay memory
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class PrioritizedReplayMemory:
    """Prioritized Experience Replay memory for more efficient learning"""

    def __init__(
        self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000
    ):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Importance sampling start value
        self.beta_end = beta_end  # Importance sampling end value
        self.beta_frames = beta_frames
        self.frame = 1  # Current frame counter
        self.epsilon = 1e-6  # Small constant to ensure non-zero priority

    def push(self, *args):
        """Save a transition with max priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append(Transition(*args))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        """Sample a batch of transitions based on priority"""
        if len(self.memory) == 0:
            return None

        # Calculate current beta for importance sampling
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(
            1.0, self.frame / self.beta_frames
        )
        self.frame += 1

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        p = priorities**self.alpha
        p = p / p.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=p)

        # Calculate importance sampling weights
        weights = (len(self.memory) * p[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize weights

        # Get selected transitions
        transitions = [self.memory[idx] for idx in indices]
        batch = Transition(*zip(*transitions))

        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """DQN Agent with improved training features"""

    def __init__(
        self,
        learning_rate=0.0003,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=15000,
        memory_size=100000,
        batch_size=128,
        target_update=500,
        prioritized_replay=True,
        double_dqn=True,
        gradient_clipping=10.0,
        model_dir="models",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize policy and target networks
        self.policy_net = EnhancedDQN().to(self.device)
        self.target_net = EnhancedDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network always in evaluation mode

        # Training parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0

        # Advanced training techniques
        self.prioritized_replay = prioritized_replay
        self.double_dqn = double_dqn
        self.gradient_clipping = gradient_clipping

        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Initialize replay memory
        if prioritized_replay:
            self.memory = PrioritizedReplayMemory(memory_size)
        else:
            self.memory = deque([], maxlen=memory_size)

        # Create model directory if it doesn't exist
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def select_action(self, state, valid_moves):
        """Select action using epsilon-greedy policy"""
        # Convert state to torch tensor
        state_tensor = self.policy_net.state_to_input(state).to(self.device)

        # Calculate current epsilon
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

        self.steps_done += 1

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Random action
            return np.random.choice(valid_moves)
        else:
            # Greedy action based on Q-values
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)

                # Create a mask for valid actions
                mask = torch.ones(49) * float("-inf")
                mask[valid_moves] = 0

                # Apply mask to q_values
                masked_q_values = q_values + mask.to(self.device)

                # Select action with highest Q-value
                action = masked_q_values.max(1)[1].item()
                return action

    def select_action_target(self, state, valid_moves):
        """Select action using target network (used in self-play)"""
        # Convert state to torch tensor
        state_tensor = self.target_net.state_to_input(state).to(self.device)

        # Small exploration chance
        if random.random() < 0.1:
            return np.random.choice(valid_moves)

        # Greedy action based on target network Q-values
        with torch.no_grad():
            q_values = self.target_net(state_tensor)

            # Create a mask for valid actions
            mask = torch.ones(49) * float("-inf")
            mask[valid_moves] = 0

            # Apply mask to q_values
            masked_q_values = q_values + mask.to(self.device)

            # Select action with highest Q-value
            action = masked_q_values.max(1)[1].item()
            return action

    def select_action_with_model(self, state, valid_moves, model):
        """Select action using a specific model (used for playing against previous versions)"""
        # Convert state to torch tensor
        state_tensor = model.state_to_input(state).to(self.device)

        # Small exploration chance
        if random.random() < 0.1:
            return np.random.choice(valid_moves)

        # Greedy action based on provided model's Q-values
        with torch.no_grad():
            q_values = model(state_tensor)

            # Create a mask for valid actions
            mask = torch.ones(49) * float("-inf")
            mask[valid_moves] = 0

            # Apply mask to q_values
            masked_q_values = q_values + mask.to(self.device)

            # Select action with highest Q-value
            action = masked_q_values.max(1)[1].item()
            return action

    def optimize_model(self):
        """Perform one step of optimization"""
        # Check if enough samples in memory
        if len(self.memory) < self.batch_size:
            return None

        # Sample transitions from memory
        if self.prioritized_replay:
            batch_data = self.memory.sample(self.batch_size)
            if batch_data is None:
                return None

            transitions, indices, weights = batch_data
        else:
            transitions = random.sample(self.memory, self.batch_size)
            weights = torch.ones(self.batch_size)

        # Unpack batch
        batch = Transition(*zip(*transitions))

        # Create non-final mask
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        # Create tensor batches
        state_batch = torch.cat(
            [self.policy_net.state_to_input(s) for s in batch.state]
        ).to(self.device)
        action_batch = torch.tensor([[a] for a in batch.action], device=self.device)
        reward_batch = torch.tensor([r for r in batch.reward], device=self.device)

        # Get current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Initialize next state values to zero
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        # Process non-final next states
        if any(non_final_mask):
            non_final_next_states = torch.cat(
                [
                    self.policy_net.state_to_input(s)
                    for s in batch.next_state
                    if s is not None
                ]
            ).to(self.device)

            if self.double_dqn:
                # Double DQN: actions selected by policy net, values from target net
                with torch.no_grad():
                    # Get actions from policy network
                    policy_output = self.policy_net(non_final_next_states)
                    best_actions = policy_output.max(1)[1].unsqueeze(1)

                    # Get values from target network
                    next_state_values_temp = self.target_net(non_final_next_states)
                    next_state_values[non_final_mask] = next_state_values_temp.gather(
                        1, best_actions
                    ).squeeze(1)
            else:
                # Regular DQN
                with torch.no_grad():
                    next_state_values[non_final_mask] = self.target_net(
                        non_final_next_states
                    ).max(1)[0]

        # Calculate expected Q values
        expected_q_values = reward_batch + (
            self.gamma
            * next_state_values
            * (1 - torch.tensor(batch.done, device=self.device))
        )

        # Calculate loss
        # Huber loss combines MSE for small errors and MAE for large errors (more robust)
        loss = F.smooth_l1_loss(
            current_q_values, expected_q_values.unsqueeze(1), reduction="none"
        )

        # Apply importance sampling weights for prioritized replay
        if self.prioritized_replay:
            # Weight the loss
            loss = (loss * weights.unsqueeze(1).to(self.device)).mean()

            # Update priorities
            td_errors = (
                torch.abs(expected_q_values - current_q_values.squeeze())
                .detach()
                .cpu()
                .numpy()
            )
            self.memory.update_priorities(indices, td_errors)
        else:
            loss = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.gradient_clipping
        )

        self.optimizer.step()

        # Update learning rate based on loss
        self.scheduler.step(loss)

        return loss.item()

    def update_target_network(self):
        """Update target network using polyak averaging for smoother updates"""
        tau = 0.01  # Small value for soft update
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1 - tau) * target_param.data
            )

    def hard_update_target_network(self):
        """Hard update target network (copy all weights)"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filename):
        """Save model to file"""
        filepath = os.path.join(self.model_dir, filename)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "steps_done": self.steps_done,
            },
            filepath,
        )

    def load_model(self, filename):
        """Load model from file"""
        filepath = os.path.join(self.model_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.steps_done = checkpoint["steps_done"]

    def get_model_copy(self):
        """Create a copy of the current policy network"""
        model_copy = EnhancedDQN().to(self.device)
        model_copy.load_state_dict(self.policy_net.state_dict())
        model_copy.eval()
        return model_copy


# Add missing math import
import math

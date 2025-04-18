# ai/dqn_agent.py
import numpy as np
import random
from collections import deque
from .network import DQNNetwork
from .memory import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size=49, action_size=49, hidden_layers=[128, 64], learning_rate=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                 memory_size=10000, batch_size=64, target_update_freq=1000):
        """
        Initialize DQN Agent with parameters
        
        Args:
            state_size (int): Size of state space (board positions)
            action_size (int): Size of action space (possible moves)
            hidden_layers (list): Sizes of hidden layers in neural network
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor for future rewards
            epsilon (float): Exploration rate
            epsilon_decay (float): Rate at which epsilon decays
            epsilon_min (float): Minimum exploration rate
            memory_size (int): Size of replay buffer
            batch_size (int): Size of training batch
            target_update_freq (int): How often to update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        # Create main and target networks
        self.model = DQNNetwork(state_size, action_size, hidden_layers, learning_rate)
        self.target_model = DQNNetwork(state_size, action_size, hidden_layers, learning_rate)
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from model to target_model"""
        self.target_model.model.set_weights(self.model.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
        
    def act(self, state, valid_actions=None, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state (np.array): Current state
            valid_actions (list): List of valid actions
            training (bool): Whether agent is in training mode
        
        Returns:
            int: Chosen action
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
            
        if len(valid_actions) == 0:
            return None
            
        # Explore: choose a random action
        if training and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Exploit: choose best action based on Q-values
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state)[0]
        
        # Mask invalid actions with very negative values
        masked_q_values = np.full(self.action_size, -np.inf)
        for action in valid_actions:
            masked_q_values[action] = q_values[action]
        
        return np.argmax(masked_q_values)
    
    def replay(self):
        """Train model on batch of experiences from replay buffer"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Get current Q-values
        current_q = self.model.predict(states)
        
        # Get next Q-values from target network
        next_q = self.target_model.predict(next_states)
        
        # Calculate target Q-values
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Train network
        loss = self.model.train(states, target_q)
        
        # Update target network if needed
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def load(self, name):
        """Load model weights from file"""
        self.model.load(name)
        self.update_target_network()
         
    def save(self, name):
        """Save model weights to file with automatic .weights.h5 extension handling"""
        self.model.save(name)  # Delegate saving to DQNNetwork class
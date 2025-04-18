# ai/network.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNNetwork:
    def __init__(self, state_size, action_size, hidden_layers=[128, 64], learning_rate=0.001):
        """
        Initialize DQN network with parameters
        
        Args:
            state_size (int): Size of state input (board positions)
            action_size (int): Size of action output (possible moves)
            hidden_layers (list): Sizes of hidden layers
            learning_rate (float): Learning rate for optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model(hidden_layers)
        
    def _build_model(self, hidden_layers):
        """Build and compile neural network model"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], input_dim=self.state_size, activation='relu'))
        
        # Hidden layers
        for size in hidden_layers[1:]:
            model.add(Dense(size, activation='relu'))
        
        # Output layer (Q-values for each action)
        model.add(Dense(self.action_size, activation='linear'))
        
        # Compile model
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def predict(self, state):
        """Predict Q-values for given state"""
        return self.model.predict(state, verbose=0)
    
    def train(self, states, target_q):
        """Train network on batch of states and target Q-values"""
        history = self.model.fit(states, target_q, epochs=1, verbose=0)
        return history.history['loss'][0]
    
    def save(self, name):
        """Save model weights with automatic .weights.h5 extension handling"""
        if not name.endswith('.weights.h5'):
            if name.endswith('.h5'):
                name = name.replace('.h5', '.weights.h5')
            else:
                name += '.weights.h5'
        self.model.save_weights(name)
            
    def load(self, name):
        """Load model weights (auto-handles .weights.h5 extension if needed)"""
        if not name.endswith('.weights.h5'):
            if name.endswith('.h5'):
                name = name.replace('.h5', '.weights.h5')
            else:
                name += '.weights.h5'
        self.model.load_weights(name)
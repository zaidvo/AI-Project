import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedDQN(nn.Module):
    """
    Enhanced Deep Q-Network for 7x7 Tic Tac Toe with improved pattern recognition
    """

    def __init__(self, dropout_rate=0.2):
        super(EnhancedDQN, self).__init__()
        self.board_size = 7
        # Three channels: X positions, O positions, and valid moves
        self.input_channels = 3

        # Multi-scale pattern recognition with different kernel sizes
        # First convolutional block - pattern detection
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Second convolutional block - higher-level patterns
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Parallel path for detecting line formations (larger kernel)
        self.conv_lines = nn.Conv2d(self.input_channels, 32, kernel_size=5, padding=2)
        self.bn_lines = nn.BatchNorm2d(32)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Advantage and Value streams (Dueling DQN architecture)
        # Advantage: how much better is each action compared to others
        # Value: how good is the current state overall
        self.advantage_stream = nn.Sequential(
            nn.Linear(
                128 * self.board_size * self.board_size
                + 32 * self.board_size * self.board_size,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.board_size * self.board_size),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(
                128 * self.board_size * self.board_size
                + 32 * self.board_size * self.board_size,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """
        Forward pass through the network with dual pathways and dueling architecture

        Args:
            x: Input tensor of shape (batch_size, 3, 7, 7)
                Channel 0: X positions (1 where X, 0 elsewhere)
                Channel 1: O positions (1 where O, 0 elsewhere)
                Channel 2: Valid moves (1 where valid, 0 elsewhere)

        Returns:
            Q-values for each possible action
        """
        batch_size = x.size(0)

        # Main convolutional pathway
        conv_path = F.relu(self.bn1(self.conv1(x)))
        conv_path = F.relu(self.bn2(self.conv2(conv_path)))
        conv_path = F.max_pool2d(
            conv_path, 2, stride=1, padding=1
        )  # Maintain spatial dimensions

        conv_path = F.relu(self.bn3(self.conv3(conv_path)))
        conv_path = F.relu(self.bn4(self.conv4(conv_path)))

        # Line detection pathway
        line_path = F.relu(self.bn_lines(self.conv_lines(x)))

        # Flatten both pathways
        conv_flat = conv_path.view(batch_size, -1)
        line_flat = line_path.view(batch_size, -1)

        # Concatenate features from both pathways
        combined = torch.cat([conv_flat, line_flat], dim=1)
        combined = self.dropout(combined)

        # Dueling architecture: split into advantage and value streams
        advantage = self.advantage_stream(combined)
        value = self.value_stream(combined)

        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    def state_to_input(self, state):
        """
        Convert a state (7x7 numpy array) to the input format for the network

        Args:
            state: 7x7 numpy array with 1 for X, -1 for O, 0 for empty

        Returns:
            torch.Tensor of shape (1, 3, 7, 7)
        """
        import numpy as np

        # Create three binary channels
        x_channel = (state == 1).astype(np.float32)
        o_channel = (state == -1).astype(np.float32)
        valid_moves = (state == 0).astype(np.float32)  # Empty cells are valid moves

        # Stack channels
        input_tensor = np.stack([x_channel, o_channel, valid_moves], axis=0)

        # Convert to torch tensor and add batch dimension
        return torch.FloatTensor(input_tensor).unsqueeze(0)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class AlphaZeroNetwork(nn.Module):
    """Neural network for 5x5 board with 4-in-a-row win condition"""

    def __init__(self, board_size=5, num_resblocks=5, num_channels=128):
        super(AlphaZeroNetwork, self).__init__()
        self.board_size = board_size
        action_size = board_size * board_size

        # Input layer
        self.conv_input = nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.resblocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_resblocks)]
        )

        # Policy head - predict next move probabilities
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, action_size)

        # Value head - predict game outcome
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Common layers
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.resblocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)  # Logits

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Range [-1, 1]

        return policy, value

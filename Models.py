import torch
from torch import nn as nn
from torch.nn import functional as F


class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        # Assuming the input state is a 4x4 board. Adjust according to your actual input dimensions.
        # Note: input should be (batch_size, channels, height, width). Here, channels = 1 for the game board.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=1)  # 32 filters, kernel size 2x2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)

        # Calculate the size of the output from the last conv layer to connect it to the first linear layer
        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(4))  # Assuming a 4x4 input grid
        convh = convw  # Square output
        linear_input_size = convw * convh * 64  # Output size * number of output channels from the last conv layer

        self.fc_value = nn.Linear(linear_input_size, 1)
        self.fc_advantage = nn.Linear(linear_input_size, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        advantage_mean = torch.mean(advantage, dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean
        return q_values

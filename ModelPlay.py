import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from Game import Game
import matplotlib.pyplot as plt




# Define the model class as before
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

# Assuming the Game class has a render method to visualize the game state
# Load your trained model
def load_model(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to play and render a game
def play_and_render_game(model, game, num_steps=None):
    state = game.reset()
    done = False
    if num_steps is None:
        while not done:
            state_tensor = torch.FloatTensor(state['board']).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = q_values.max(1)[1].item()

            next_state, reward, done = game.step(action)
            game.render()
            time.sleep(0.5)  # Sleep for half a second
            state = next_state
    else:
        for step in range(num_steps):
            # Assuming state is a dictionary with 'board' key holding the game state
            state_tensor = torch.FloatTensor(state['board']).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = q_values.max(1)[1].item()

            next_state, reward, done = game.step(action)
            game.render()  # Render the game; implement this method in Game class

            if done:
                break
            state = next_state
            time.sleep(0.5)  # Sleep for half a second

# Paths to the model and the game environment
if __name__ == '__main__':
    if sys.argv[1]:
        model_path = sys.argv[1]
    else:
        print("Please provide the path to the model file")
        exit()
    random.seed()
    game_env = Game()  # Initialize your game environment

    # Load the model
    model = load_model(model_path, DuelingDQN)

    # Play the game and render
    play_and_render_game(model, game_env)

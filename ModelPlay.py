import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from Game import Game
import matplotlib.pyplot as plt

# Define the model class as before
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_advantage = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
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
def play_and_render_game(model, game, num_steps=200):
    state = game.reset()
    for step in range(num_steps):
        # Assuming state is a dictionary with 'board' key holding the game state
        state_tensor = torch.FloatTensor(state['board']).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.max(1)[1].item()

        next_state, reward, done = game.step(action)
        game.render()  # Render the game; implement this method in Game class

        if done:
            break
        state = next_state

# Paths to the model and the game environment
if __name__ == '__main__':
    if sys.argv[1]:
        model_path = sys.argv[1]
    else:
        print("Please provide the path to the model file")
        exit()
    game_env = Game()  # Initialize your game environment

    # Load the model
    model = load_model(model_path, DuelingDQN)

    # Play the game and render
    play_and_render_game(model, game_env)

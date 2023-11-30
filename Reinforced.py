from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # change this line
import tqdm
from matplotlib import pyplot as plt
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn, optim
from torch.distributions import Categorical

import math
import time
import random
from pynput import keyboard

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
from Game import Game
from Cell import Cell, EmptyCell

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine = nn.Linear(16, 128)
        self.drop = nn.Dropout(p=0.6)
        self.action_head = nn.Linear(128, 4)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine(x))
        x = self.drop(x)
        action_scores = self.action_head(x)
        return F.softmax(action_scores, dim=-1)

env = Game()
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state: dict):
    flat_board = np.array(state['board']).flatten()  # Flatten the board
    state_tensor = torch.FloatTensor(flat_board)  # Convert numpy array to Tensor
    state_tensor = state_tensor.unsqueeze(0)
    probs = policy(state_tensor)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + 0.99 * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    running_reward = 10
    for i_episode in range(10000):
        env.reset()
        state = env.getState()  # This will be a dictionary
        episode_reward = 0
        rewards = []
        t = 0
        win_flag = False
        while (True): #for t in range(10000):  # Don't infinite loop while learning
            t += 1
            action = select_action(state)
            next_state = env.step(action)  # Assuming this returns a dict {"reward": _, "done": _}
            score_delta = next_state['score'] - state['score']
            max_value_delta = next_state['max_value'] - state['max_value']
            total_value_delta = next_state['total_value'] - state['total_value']
            total_empty_cells_delta = state['total_empty_cells'] - next_state['total_empty_cells']
            if next_state['win'] and not win_flag:
                win_flag = True
                win_value = 1000
                loss_value = 0
            elif next_state['game_over']:
                loss_value = -1000
                win_value = 0
            else:
                win_value = 0
                loss_value = 0
            if next_state['win']:
                print("The agent won!")
                with open("./Outputs/wins.txt", "a+") as f:
                    f.write(f"Episode {i_episode} won in {t} steps\n"+str(env.board)+"\n"+str(env.score)+"\n")
                step_reward = 10
            else:
                step_reward = 0
            reward = score_delta + max_value_delta + total_value_delta + total_empty_cells_delta + win_value + loss_value + step_reward
            episode_reward += reward
            policy.rewards.append(reward)
            done = next_state["game_over"]
            if done:
                break
            state = next_state  # move to next state
        rewards.append(running_reward)
        running_reward = running_reward * 0.99 + t * 0.01
        if i_episode % 50 == 0:
            torch.save(policy.state_dict(), "./Outputs/policy.pth")
        if i_episode % 1000 == 0:
            plt.plot(rewards)
            plt.title('Running rewards after episode ' + str(i_episode))
            plt.xlabel('Episode')
            plt.ylabel('Running reward')
            plt.savefig(f'./Outputs/rewards_{datetime.now().strftime("%Y%m%d-%H%M")}_{i_episode}.png')
        finish_episode()
        print(
        f"Episode {i_episode} finished after {t} steps with reward {episode_reward}. Running reward: {running_reward}")

def play_game(policy_path):
    # Load the policy from the saved state_dict
    policy = Policy()  # Initialize policy the same way as you did during training
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()  # Set the policy to evaluation mode

    env = Game()  # Initialize the environment

    for i_episode in range(10):  # Play 10 episodes, change it as per your requirement
        state = env.reset()
        for t in range(100):  # Number of steps in each episode. Modify accordingly
            env.render()  # This is for visualization
            action = select_action(state, policy)
            state, reward, done, _ = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

if __name__ == '__main__':
    main()

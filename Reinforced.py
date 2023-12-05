from datetime import datetime
import os

import numpy as np
import torch
from typing import Optional

import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as lr_scheduler


import math
import time
import random

from Game import Game

print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        """
        Policy Inputs:
          - 16 Squares
            1. Value of the cell
            2. Boolean if the cell to the right is the same value
            3. Boolean if the cell below is the same value
          - Last action taken
        

        """
        self.affine1 = nn.Linear(80+4, 128)
        self.affine2 = nn.Linear(128, 128)
        self.affine3 = nn.Linear(128, 128)
        self.affine4 = nn.Linear(128, 64)
        self.drop = nn.Dropout(p=0.6)
        self.action_head = nn.Linear(64, 4)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        x = self.drop(x)
        action_scores = self.action_head(x)
        # Stabilizing before softmax
        max_scores = torch.max(action_scores, dim=1, keepdim=True)[0]
        stabilized_scores = action_scores - max_scores
        return F.softmax(stabilized_scores, dim=-1)

BATCH_SIZE = 50 # Number of episodes to play before updating the model
NUM_EPISODES = 100000 # Number of episodes to play
SEED = None 
epsilon = 0.5 # Exploration rate
epsilon_decay_rate = 0.995 # Exponential decay rate for exploration prob
min_epsilon = 0.01  # Minimum epsilon value
max_epsilon = 1.0  # Maximum epsilon value
gamma = 0.01  # Discounting rate
alpha = 0.1  # Learning rate



env = Game()
policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=alpha)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5000, verbose=True, eps=NUM_EPISODES)


def one_hot_encode_action(action, num_actions=4):
    # Create a one-hot encoded vector for the action
    action_one_hot = [0] * num_actions
    if action is not None:
        action_one_hot[action] = 1
    return action_one_hot

def select_action(state: dict, last_action: int, action_policy: Optional[Policy] = policy, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice([0, 1, 2, 3])  # Randomly select an action
    flat_board = [cell for row in state['board'] for cell in row]
    # Include the last action in the state
    last_action_one_hot = one_hot_encode_action(last_action)
    state_tensor = torch.FloatTensor(flat_board + last_action_one_hot).unsqueeze(0).to(device)
    probs = action_policy(state_tensor)
    m = Categorical(probs)
    action = m.sample()
    action_policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def normalize_rewards(rewards, gamma=gamma):
    discounted_rewards = []
    running_add = 0
    for r in reversed(rewards):
        running_add = running_add * gamma + r
        discounted_rewards.insert(0, running_add)

    # Normalize the rewards
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + np.finfo(np.float32).eps.item())
    
    return discounted_rewards

def finish_episode(rewards):
    policy_loss = []
    returns = rewards.clone().detach()
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()

    if policy_loss:  # Check if policy_loss is not empty
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    del policy.rewards[:]
    del policy.saved_log_probs[:]



def decay_epsilon(epsilon = epsilon, decay_rate = epsilon_decay_rate, min_epsilon = min_epsilon):
    epsilon = max(epsilon * decay_rate, min_epsilon)
    return epsilon

def log_scale(value):
    if value <= 0:
        return 0
    else:
        return math.log2(value)
    
def calculate_reward(state, next_state):
    # Adjust these weights as needed
    score_weight = 1.0
    max_value_weight = 1.0
    total_value_weight = 1.0

    score_delta = log_scale(next_state['score']) - log_scale(state['score'])
    max_value_delta = log_scale(next_state['max_value']) - log_scale(state['max_value'])
    total_value_delta = log_scale(next_state['total_value']) - log_scale(state['total_value'])

    reward = (score_weight * score_delta +
              max_value_weight * max_value_delta +
              total_value_weight * total_value_delta)

    return reward

def main():
    global SEED
    global epsilon
    running_reward = 0
    rewards = []
    max_tiles = []
    non_repeating_moves = []
    repeating_moves = []
    batch_policy_loss = []
    
    #Move everything from Outputs into Outputs/Archive/{datetime}
    date = datetime.now().strftime('%Y%m%d-%H%M')
    if len(os.listdir("./Outputs/")) > 1:
        
        os.mkdir(f"./Outputs/Archive/{date}")
        for file in os.listdir("./Outputs"):
            if file != "Archive":
                if os.name == 'nt':
                    os.system(f"move .\\Outputs\\{file} .\\Outputs\\Archive\\{date}")
                elif os.name == 'posix':
                    os.system(f"mv ./Outputs/{file} ./Outputs/Archive/{date}/{file}")

    for i_episode in range(NUM_EPISODES):
        last_action = -1
        env.reset(seed=SEED)
        epsilon = decay_epsilon()  
        state = env.getState()
        episode_reward = 0
        
        t = 0
        win_flag = False
        same_count = 0
        num_repeating_moves = 0
        num_non_repeating_moves = 0
        while (True): 
            t += 1
            action = select_action(state, last_action, epsilon=epsilon)
            
            next_state = env.step(action)

            # REWARD WEIGHTS
            score_weight = 0
            max_value_weight = 0.0
            total_value_weight = 0.0
            total_empty_cells_weight = 1.0
            win_loss_weight = 0.0
            total_step_weight = 1.0
            
            # Score delta
            if next_state['score'] > state['score'] and state['score'] > 0:
                score_delta = next_state['score'] - state['score']
            elif next_state['score'] > state['score'] and state['score'] == 0:
                score_delta = math.log(next_state['score'], 2)
            else:
                score_delta = 0

            # Max value delta
            if next_state['max_value'] > state['max_value']:
                max_value_delta = math.log2(next_state['max_value']) - math.log2(state['max_value'])
            else:
                max_value_delta = 0

            # Total value delta
            if next_state['total_value'] > state['total_value']:
                total_value_delta = math.log(next_state['total_value'], 2) - math.log(state['total_value'], 2)
            else:
                total_value_delta = 0

            # Total empty cells delta
            if next_state['total_empty_cells'] > state['total_empty_cells']:
                total_empty_cells_delta = next_state['total_empty_cells'] - state['total_empty_cells']
                same_count = 0
            elif next_state['total_empty_cells'] == state['total_empty_cells'] and action == last_action:
                same_count += 1
                if same_count > 0:
                    total_empty_cells_delta = -1
                else:
                    total_empty_cells_delta = 0
            else:
                total_empty_cells_delta = 0
                same_count = 0

            if action == last_action:
                num_repeating_moves += 1
            else:
                num_non_repeating_moves += 1
            # Win/Loss Reward
            if next_state['win'] and not win_flag:
                win_flag = True
                win_value = 0 #10
                loss_value = 0
                print("The agent won!")
                with open("./Outputs/wins.txt", "a+") as f:
                    f.write(f"Episode {i_episode} won in {t} steps\n"+str(env.board)+"\n"+str(env.score)+"\n")
            elif next_state['game_over']:
                loss_value = 0 #-5
                win_value = 0
            else:
                win_value = 0
                loss_value = 0

            # Step reward
            if next_state['score'] > state['score']:
                step_reward = 1
            else:
                step_reward = -1

            reward = (score_weight * score_delta +
                        max_value_weight * max_value_delta +
                        total_value_weight * total_value_delta +
                        total_empty_cells_weight * total_empty_cells_delta +
                        win_value +
                        loss_value * win_loss_weight +
                        step_reward * total_step_weight)
            episode_reward += reward
            policy.rewards.append(reward)
            done = next_state["game_over"]
            if done:
                break
            last_action = action
            state = next_state  # move to next state
        
        # Update the learning rate
        scheduler.step(episode_reward)
        
        # Compute the policy loss for the current episode
        returns = normalize_rewards(policy.rewards)
        episode_policy_loss = []
        for log_prob, R in zip(policy.saved_log_probs, returns):
            episode_policy_loss.append(-log_prob * R)
        
        # Accumulate policy loss over multiple episodes
        batch_policy_loss.extend(episode_policy_loss)
        
        # Update model parameters after every BATCH_SIZE episodes
        if i_episode % BATCH_SIZE == 0 or i_episode == NUM_EPISODES - 1:
            optimizer.zero_grad()
            batch_loss = torch.cat(batch_policy_loss).sum()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            batch_policy_loss = []  # Reset for the next batch
            #SEED = random.randint(0, 1000000)

        # Clear the rewards and saved log probabilities
        del policy.rewards[:]
        del policy.saved_log_probs[:]
        rewards.append(episode_reward/t)
        max_tiles.append(state['max_value'])
        non_repeating_moves.append(num_non_repeating_moves)
        repeating_moves.append(num_repeating_moves)
        
        running_reward = np.mean(rewards[-100:])
        if i_episode % 5 == 0:
            print(
                f"Episode: {str(i_episode).ljust(7)} | Steps: {str(t).ljust(5)} | Reward: {str(episode_reward).ljust(6)} | Average Points Per Step: {f'{episode_reward/t:.5}'.ljust(10)} | Running reward: {f'{running_reward:.5}'.ljust(10)}".center(10, " "))
        if i_episode % 50 == 0:
            #SEED = random.randint(0, 1000000)
            torch.save(policy.state_dict(), f"./Outputs/policy_{date}.pth")
        if i_episode % 1000 == 0 and i_episode > 0:
            x = [i for i in range(len(rewards))]
            rewards_coeff = np.polyfit(x, rewards, 3)
            plt.figure(figsize=(20, 10))
            plt.scatter(x, rewards)
            plt.plot(x, np.polyval(rewards_coeff, x), color="red")
            plt.title('Running rewards after episode ' + str(i_episode))
            plt.xlabel('Episode')
            plt.ylabel('Running reward')
            plt.savefig(f'./Outputs/rewards_{datetime.now().strftime("%Y%m%d-%H%M")}_{i_episode}.png')
            plt.close()
            max_tiles_coeff = np.polyfit(x, max_tiles, 3)
            plt.figure(figsize=(20, 10))
            plt.scatter(x, max_tiles)
            plt.plot(x, np.polyval(max_tiles_coeff, x), color="red")
            plt.title('Max tiles after episode ' + str(i_episode))
            plt.xlabel('Episode')
            plt.ylabel('Max tile')
            plt.savefig(f'./Outputs/max_tiles_{datetime.now().strftime("%Y%m%d-%H%M")}_{i_episode}.png')
            plt.close()
            non_repeating_moves_coeff = np.polyfit(x, non_repeating_moves, 3)
            plt.figure(figsize=(20, 10))
            plt.scatter(x, non_repeating_moves)
            plt.plot(x, np.polyval(non_repeating_moves_coeff, x), color="red")
            plt.title('Non-repeating moves after episode ' + str(i_episode))
            plt.xlabel('Episode')
            plt.ylabel('Non-repeating moves')
            plt.savefig(f'./Outputs/non_repeating_moves_{datetime.now().strftime("%Y%m%d-%H%M")}_{i_episode}.png')
            plt.close()
            repeating_moves_coeff = np.polyfit(x, repeating_moves, 3)
            plt.figure(figsize=(20, 10))
            plt.scatter(x, repeating_moves)
            plt.plot(x, np.polyval(repeating_moves_coeff, x), color="red")
            plt.title('Repeating moves after episode ' + str(i_episode))
            plt.xlabel('Episode')
            plt.ylabel('Repeating moves')
            plt.savefig(f'./Outputs/repeating_moves_{datetime.now().strftime("%Y%m%d-%H%M")}_{i_episode}.png')
        if i_episode % 10000 == 0 and i_episode > 0:
            play_game(f"./Outputs/policy_{date}.pth")
        finish_episode(returns)
        

def play_game(policy_path):
    policy = Policy()
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.to(device).eval()


    env = Game()

    env.reset()
    state = env.getState()
    last_action = -1
    play_ep = 0.0
    while not env.game_over():
        env.render()
        action = select_action(state, last_action, policy, epsilon=play_ep)
        if action == last_action:
            play_ep += 0.1
        else:
            play_ep = 0.0
        last_action = action
        next_state = env.step(action)
        if next_state['game_over']:
            print("\33[41mLOSER AI\33[0m".center(20))
        time.sleep(.5)
        state = next_state
    env.render()


if __name__ == '__main__':
    main()
    #play_game("./Outputs/policy.pth")
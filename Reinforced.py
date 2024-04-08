import math
import os
from time import gmtime, strftime

import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from Game import Game
import matplotlib.pyplot as plt


curr_dt = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
if not os.path.exists("Outputs"):
    os.makedirs("Outputs")

if not os.path.exists("Models"):
    os.makedirs("Models")

if not os.path.exists(f"Outputs/{curr_dt}"):
    os.makedirs(f"Outputs/{curr_dt}")

if not os.path.exists(f"Models/{curr_dt}"):
    os.makedirs(f"Models/{curr_dt}")

if not os.path.exists(f"Outputs/{curr_dt}/images"):
    os.makedirs(f"Outputs/{curr_dt}/images")

if not os.path.exists(f"Outputs/{curr_dt}/average_scores.csv"):
    with open(f"Outputs/{curr_dt}/average_scores.csv", "w") as f:
        f.write("phase,average_score\n")





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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, model):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount rate
        self.model = model()
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer(10000)

    def act(self, state, test=False):
        if not test and random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            action = q_values.max(1)[1].item()
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            action = q_values.max(1)[1].item()
        return action

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.model(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        #agent.epsilon = self.epsilon_min + (1 - self.epsilon_min) * math.exp(
        #    -1. * episode / 500)
        loss = F.mse_loss(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def push_replay_buffer(self, *args):
        self.replay_buffer.push(*args)

def test_agent(game, agent, num_tests=10):
    total_scores = []
    for test in range(num_tests):
        state = game.reset()  # Reset game to the initial state
        score = 0
        done = False
        prev_action = None
        count = 0
        while not done:
            with torch.no_grad():
                action = agent.act(state['board'], test=True)
                if prev_action == action:
                    count += 1
                    if count > 20:
                        break
                else:
                    count = 0
                prev_action = action
            next_state, reward, done = game.step(action)
            score += reward
            state = next_state
        total_scores.append(score)
        print(f"Test Game: {test + 1}, Score: {score}")
    average_score = sum(total_scores) / num_tests
    print(f"Average Score over {num_tests} tests: {average_score}")
    return average_score

# Initialize game and agent
game = Game()
average_scores = pandas.Series()
state_size = 16  # 4x4 grid flattened
action_size = 4  # left, right, up, down
batch_size = 64
num_tests = 10
agent = DQNAgent(state_size, action_size, batch_size, DuelingDQN)

# Main loop for multiple training and testing phases
last_average_score = None  # Track the last average score for performance comparison
num_episodes_per_phase = 100
total_reward = 0

for phase in range(10000):  # Three phases of training and testing
    print(f"--- Starting Training Phase {phase + 1} ---")
    state = game.getRandomState()
    total_reward = 0
    for episode in range(num_episodes_per_phase):
        tgame = Game(state=state)
        for _ in range(1):
            action = agent.act(state['board'])
            next_state, reward, done = tgame.step(action)
            agent.push_replay_buffer(state['board'], action, reward, next_state['board'], done)
            agent.learn()
            state = next_state
            total_reward += reward
            if done:
                break

        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    if phase % 100 == 0:
        torch.save(agent.model.state_dict(), f"Models/{curr_dt}/model_phase_{phase + 1}.pth")
    print(f"--- Starting Testing Phase after Training Phase {phase + 1} ---")
    average_score = test_agent(game, agent, num_tests)
    with open(f"Outputs/{curr_dt}/average_scores.csv", "a") as f:
        f.write(f"{phase+1},{average_score}\n")
    average_scores = pandas.read_csv(f"Outputs/{curr_dt}/average_scores.csv", index_col=0, header=0)
    plt.figure(figsize=(10, 6))
    plt.plot(average_scores.index, average_scores['average_score'], marker='o', linestyle='-', color='blue')
    z = np.polyfit(average_scores.index, average_scores['average_score'], 1)
    p = np.poly1d(z)

    # add trendline to plot
    plt.plot(average_scores.index, p(average_scores.index))
    plt.title('Average Scores Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True)
    plt.savefig(f"Outputs/{curr_dt}/images/{phase + 1}.png")
    plt.close()

    # Dynamic Epsilon and Learning Rate Adjustments after Testing
    # Decay epsilon every N episodes
    # Example logic to adjust epsilon based on performance
    performance_threshold = 50
    if average_score > performance_threshold:
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    else:
        agent.epsilon = min(1.0, agent.epsilon * 1.01)  # Slightly increase epsilon if performance is below threshold
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = max(1e-5, param_group['lr'] * 0.95) if last_average_score is not None and average_score <= last_average_score else param_group['lr']
    last_average_score = average_score

    print(f"End of Phase {phase + 1} - Adjusted Epsilon: {agent.epsilon}, Learning Rate: {agent.optimizer.param_groups[0]['lr']}")

print("--- Final Testing ---")
torch.save(agent.model.state_dict(), f"Models/{curr_dt}/final_model.pth")
test_agent(game, agent, num_tests=20)

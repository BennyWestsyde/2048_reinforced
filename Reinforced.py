import math
import os
import threading
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

if not os.path.exists(f"Outputs/{curr_dt}/images/average_scores"):
    os.makedirs(f"Outputs/{curr_dt}/images/average_scores")

if not os.path.exists(f"Outputs/{curr_dt}/images/high_tiles"):
    os.makedirs(f"Outputs/{curr_dt}/images/high_tiles")

if not os.path.exists(f"Outputs/{curr_dt}/images/moves_before_break"):
    os.makedirs(f"Outputs/{curr_dt}/images/moves_before_break")

if not os.path.exists(f"Outputs/{curr_dt}/average_scores.csv"):
    with open(f"Outputs/{curr_dt}/average_scores.csv", "w") as f:
        f.write("phase,average_score\n")

if not os.path.exists(f"Outputs/{curr_dt}/high_tiles.csv"):
    with open(f"Outputs/{curr_dt}/high_tiles.csv", "w") as f:
        f.write("phase,high_tile\n")

if not os.path.exists(f"Outputs/{curr_dt}/moves_before_break.csv"):
    with open(f"Outputs/{curr_dt}/moves_before_break.csv", "w") as f:
        f.write("phase,moves_before_break\n")


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
        self.epsilon_max = 1.0
        self.epsilon = self.epsilon_max
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate_decay = 0.999
        self.learning_rate_min = 1e-5
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
    total_high_tile = []
    total_moves_before_break = []
    break_count = 0
    for test in range(num_tests):
        state = game.reset()  # Reset game to the initial state
        score = 0
        done = False
        prev_action = None
        count = 0
        total_actions = 0
        break_flag = False
        while not done:
            with torch.no_grad():
                action = agent.act(state['board'], test=True)
                if prev_action == action:
                    count += 1
                    if count > repetition_allowance:
                        break_flag = True
                        break
                else:
                    count = 0
                prev_action = action
            next_state, reward, done = game.step(action)
            total_actions += 1
            score += reward
            state = next_state
        total_scores.append(score)
        total_moves_before_break.append(total_actions)
        total_high_tile.append(game.max_value)
        if break_flag:
            print(f"Test Game: {test + 1}, Score: {score}, Breaking due to repetition")
            break_count += 1
        else:
            print(f"Test Game: {test + 1}, Score: {score}")
    average_score = sum(total_scores) / num_tests
    average_break_count = (break_count / num_tests) * 100
    average_high_tile = sum(total_high_tile) / num_tests
    average_moves_before_break = sum(total_moves_before_break) / num_tests

    print(f"Average Score over {num_tests} tests: {average_score}, Break Percentage: {average_break_count}%")
    return average_score, average_high_tile, average_moves_before_break


def save_plot(file):
    temp_df = pandas.read_csv(f"Outputs/{curr_dt}/{file}", index_col=0, header=0)
    temp_df.columns = ['value']
    plt.figure(figsize=(10, 6))
    plt.scatter(temp_df.index, temp_df['value'])
    z = np.polyfit(temp_df.index, temp_df['value'], 1)
    p = np.poly1d(z)

    # add trendline to plot
    plt.plot(temp_df.index, p(temp_df.index), color='red')
    if 'average_score' in file:
        plt.title('Average Scores Over Phases')
        plt.ylabel('Average Score')
        output_path = f"Outputs/{curr_dt}/images/average_scores/{phase + 1}.png"
    elif 'high_tiles' in file:
        plt.title('High Tile Over Phases')
        plt.ylabel('High Tile')
        output_path = f"Outputs/{curr_dt}/images/high_tiles/{phase + 1}.png"
    elif 'moves_before_break' in file:
        plt.title('Moves Before Break Over Phases')
        plt.ylabel('Moves Before Break')
        output_path = f"Outputs/{curr_dt}/images/moves_before_break/{phase + 1}.png"
    plt.xlabel('Episode')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def save_videos(phase):
    if phase <= 100:
        return
    average_thread = threading.Thread(target=os.system, args=(
    f"ffmpeg -r 100 -i Outputs/{curr_dt}/images/average_scores/%d.png -vcodec mpeg4 -y Outputs/{curr_dt}/average_scores_{phase}.mp4",))
    high_thread = threading.Thread(target=os.system, args=(
    f"ffmpeg -r 100 -i Outputs/{curr_dt}/images/high_tiles/%d.png -vcodec mpeg4 -y Outputs/{curr_dt}/high_tiles_{phase}.mp4",))
    moves_thread = threading.Thread(target=os.system, args=(
    f"ffmpeg -r 100 -i Outputs/{curr_dt}/images/moves_before_break/%d.png -vcodec mpeg4 -y Outputs/{curr_dt}/moves_before_break_{phase}.mp4",))
    average_thread.start()
    high_thread.start()
    moves_thread.start()
    average_thread.join()
    high_thread.join()
    moves_thread.join()


# Initialize game and agent
game = Game()
state_size = 16  # 4x4 grid flattened
action_size = 4  # left, right, up, down
batch_size = 64
num_tests = 10
agent = DQNAgent(state_size, action_size, batch_size, DuelingDQN)
repetition_allowance = 20

# Main loop for multiple training and testing phases
last_average_score = None  # Track the last average score for performance comparison
num_phases = 10000
num_episodes_per_phase = 100
num_actions_per_episode = 1
num_outputs_per_episode = 10
save_model_interval = 100
save_plot_interval = 1
save_video_interval = num_phases // 10
performance_threshold = 20
total_reward = 0

for phase in range(num_phases):
    print(f"------ Starting Training Phase {phase + 1} ------")
    state = game.getRandomState()
    for episode in range(num_episodes_per_phase):
        temp_game = Game(state=state)
        for _ in range(num_actions_per_episode):
            action = agent.act(state['board'])
            next_state, reward, done = temp_game.step(action)
            agent.push_replay_buffer(state['board'], action, reward, next_state['board'], done)
            agent.learn()
            state = next_state
            total_reward += reward
            if done:
                break

        if episode % (num_episodes_per_phase // num_outputs_per_episode) == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    if phase % save_model_interval == 0:
        torch.save(agent.model.state_dict(), f"Models/{curr_dt}/model_phase_{phase + 1}.pth")
    if phase % save_video_interval == 0:
        save_videos(phase)
    print(f"--- Starting Testing Phase after Training Phase {phase + 1} ---")
    average_score, high_tile, moves_before_break = test_agent(game, agent, num_tests=num_tests)

    with open(f"Outputs/{curr_dt}/average_scores.csv", "a") as f:
        f.write(f"{phase + 1},{average_score}\n")
    with open(f"Outputs/{curr_dt}/high_tiles.csv", "a") as f:
        f.write(f"{phase + 1},{high_tile}\n")
    with open(f"Outputs/{curr_dt}/moves_before_break.csv", "a") as f:
        f.write(f"{phase + 1},{moves_before_break}\n")
    if phase % save_plot_interval == 0:
        save_plot("average_scores.csv")
        save_plot("high_tiles.csv")
        save_plot("moves_before_break.csv")

    if average_score > performance_threshold:
        performance_threshold *= 1.01
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = max(agent.learning_rate_min, param_group[
                'lr'] * agent.learning_rate_decay) if last_average_score is not None and average_score <= last_average_score else \
                param_group['lr']
        print(f"Performance threshold reached, adjusted performance threshold to {performance_threshold}")
    else:
        agent.epsilon = min(agent.epsilon_max, agent.epsilon * (
                    1 / agent.epsilon_decay))  # Slightly increase epsilon if performance is below threshold

    last_average_score = average_score
    print(
        f"End of Phase {phase + 1} - Adjusted Epsilon: {agent.epsilon}, Learning Rate: {agent.optimizer.param_groups[0]['lr']}")

print("--- Final Testing ---")
torch.save(agent.model.state_dict(), f"Models/{curr_dt}/final_model.pth")
test_agent(game, agent, num_tests=20)
save_videos("final")

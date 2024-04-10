import _pickle
import math
import os
import sys
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
import subprocess
from tabulate import tabulate

from ffmpeg.asyncio import FFmpeg

print(torch.cuda.is_available())  # Returns True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs on the machine
print(torch.cuda.current_device())  # Device number of the active GPU (e.g., 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # Output: device(type='cuda') if GPU is available, otherwise 'cpu'

torch.set_default_device(device)

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    model_path = sys.argv[1]
    curr_dt = os.path.split(os.path.split(os.path.abspath(model_path))[-2])[-1]
    print(f"Loading model from {curr_dt}")
    with open(f"Outputs/{curr_dt}/average_scores.csv", "r") as f:
        phase_offset = int(f.readlines()[-1].split(",")[0])
else:
    phase_offset = 0
    curr_dt = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    model_path = None

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
    def __init__(self, state_size, action_size, batch_size, model, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.epsilon_max = 0.99999
        self.epsilon = self.epsilon_max
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate_decay = 0.999
        self.learning_rate_min = 1e-5
        self.learning_rate_max = 1e-3
        self.gamma = 0.99  # Discount rate
        self.model = model()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer(10000)

    def act(self, state, test=False):
        # Convert the state to a tensor and add a channel dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        state_tensor = state_tensor.to(device)  # Ensure tensor is on the correct device

        with torch.no_grad():  # Ensure we're in evaluation mode for inference
            q_values = self.model(state_tensor)
            action = q_values.max(1)[1].item()
        return action

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Assuming states is a list of 4x4 game boards, reshape them for the model
        states = torch.FloatTensor(states).unsqueeze(1).to(device)  # Move to the correct device
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.model(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q * (1 - dones)
        # if the last reward was over 0, we should narrow the epsilon

        #if rewards[-1] > 0:
        #    agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.9999)
        #else:
        #    agent.epsilon = min(agent.epsilon_max, agent.epsilon * 1.00001)
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
    game_outputs = []
    breaks = []
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
                    rep_flag = True
                    count += 1
                    if count > repetition_allowance:
                        break_flag = True
                        break
                else:
                    count = 0
                prev_action = action
            next_state, reward, done = game.step(action)
            breaks.append(1 if break_flag else 0)
            total_actions += 1
            score += reward
            state = next_state
        total_scores.append(score)
        total_moves_before_break.append(total_actions)
        total_high_tile.append(game.max_value)
        if break_flag:

            break_count += 1
            game_outputs.append(2)
        elif game.checkWin():
            game_outputs.append(1)
        else:
            game_outputs.append(0)

    # Prepare data for tabulation
    test_results = []
    for i, (score, high_tile, moves_before_break, game_output) in enumerate(
            zip(total_scores, total_high_tile, total_moves_before_break, game_outputs)):
        test_results.append([i + 1, score, high_tile, moves_before_break, "\033[30;102mWin\033[0m" if game_output == 1 else "\033[30;101mLose\033[0m" if game_output == 0 else "\033[30;103mBreak\033[0m"])

    # Print test results using tabulate
    print(tabulate(test_results, headers=['Game', 'Score', 'High Tile', 'Moves', 'Game Output']))

    average_score = sum(total_scores) / num_tests
    average_break_count = (break_count / num_tests) * 100
    average_high_tile = sum(total_high_tile) / num_tests
    average_moves_before_break = sum(total_moves_before_break) / num_tests
    average_points_per_move = average_score / average_moves_before_break

    return average_score, average_high_tile, average_moves_before_break


def save_plot(file):
    # Assuming 'curr_dt' and 'phase' are defined outside this function and passed as arguments
    temp_df = pandas.read_csv(f"Outputs/{curr_dt}/{file}.csv", index_col=0)
    temp_df.columns = ['value']
    plt.figure(figsize=(10, 6))
    plt.scatter(temp_df.index, temp_df['value'])

    # Determine the degree for the polynomial fit
    # Minimum degree is 1 (linear), and maximum is set to 5 for practicality
    num_points = len(temp_df.index)
    degree = 1
    degree_up = 4
    while num_points//degree_up > 0:
        degree = min(max(1, num_points // degree_up), 10)  # Example dynamic degree adjustment
        degree_up = degree_up**2

    if num_points > 1:
        z = np.polyfit(temp_df.index.astype(float), temp_df['value'], degree)
        p = np.poly1d(z)
        # Add trendline to plot, ensuring index is handled as float for large datasets
        plt.plot(temp_df.index, p(temp_df.index.astype(float)), color='red')
        if degree >= 2:
            z2 = np.polyfit(temp_df.index.astype(float), temp_df['value'], 1)
            p2 = np.poly1d(z2)
            plt.plot(temp_df.index, p2(temp_df.index.astype(float)), color='green', linestyle='--')

    # Define title and y-label dynamically
    title_map = {
        'average_scores': ('Average Scores Over Phases', 'Average Score'),
        'high_tiles': ('High Tile Over Phases', 'High Tile'),
        'moves_before_break': ('Moves Before Break Over Phases', 'Moves Before Break')
    }
    for key, (title, ylabel) in title_map.items():
        if key in file:
            plt.title(title)
            plt.ylabel(ylabel)
            output_path = f"Outputs/{curr_dt}/images/{key}/{phase + 1}.png"
            break
    if not output_path:
        output_path = f"Outputs/{curr_dt}/images/{file}/{phase + 1}.png"

    plt.xlabel('Episode')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def save_vid(category, curr_phase):
    output_dir = f"Outputs/{curr_dt}/images/{category}"
    num_files = curr_phase // save_plot_interval
    desired_fps = num_files // video_length

    # Calculate previous phase based on current phase and interval
    prev_phase = curr_phase - save_video_interval

    prev_video_path = f"Outputs/{curr_dt}/{category}_{prev_phase}.mp4"
    new_video_temp_path = f"Outputs/{curr_dt}/images/{category}_temp.mp4"
    final_video_path = f"Outputs/{curr_dt}/{category}_{curr_phase}.mp4"

    # Convert images to video
    cmd_convert_images = [
        "ffmpeg", "-y",
        "-r", str(max(1, num_files // video_length)),  # Avoid division by zero for frame rate
        "-i", os.path.join(output_dir, "%d.png"),
        "-vf", "scale=1280:-1",
        "-preset", "veryslow",
        "-crf", "24",
        "-pix_fmt", "yuv420p",  # For compatibility
        new_video_temp_path
    ]
    subprocess.run(cmd_convert_images, check=True)

    # Check if a previous video exists and concatenate
    if os.path.exists(prev_video_path):
        with open("concat_list.txt", "w") as f:
            f.write(f"file '{prev_video_path}'\n")
            f.write(f"file '{new_video_temp_path}'\n")

        cmd_concat = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", "concat_list.txt",
            "-c", "copy",
            final_video_path
        ]
        subprocess.run(cmd_concat, check=True)
    else:
        # No previous video, just rename temp to final
        os.rename(new_video_temp_path, final_video_path)

    # Adjust frame rate of final video, if necessary
    cmd_adjust_fps = [
        "ffmpeg", "-y",
        "-i", final_video_path,
        "-r", str(desired_fps),
        "-preset", "veryslow",
        "-crf", "24",
        "-pix_fmt", "yuv420p",
        final_video_path + "_adjusted.mp4"  # Output adjusted video
    ]
    subprocess.run(cmd_adjust_fps, check=True)

    # Optionally replace original final video with the adjusted one
    os.rename(final_video_path + "_adjusted.mp4", final_video_path)

    # Clean up temp files if needed
    os.remove(new_video_temp_path)
    os.remove("concat_list.txt")
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))


def save_videos(curr_phase):
    if curr_phase <= 100:
        return
    threading.Thread(target=save_vid, args=("average_scores", curr_phase)).start()
    threading.Thread(target=save_vid, args=("high_tiles", curr_phase)).start()
    threading.Thread(target=save_vid, args=("moves_before_break", curr_phase)).start()


def load_model(model_path_, model_class):
    tmodel = model_class()  # Instantiate the model
    tmodel.load_state_dict(torch.load(model_path_, map_location=device))  # Load the state dict
    tmodel.to(device)  # Ensure model is on the correct device
    return tmodel





# Initialize game and agent
game = Game()
state_size = 16  # 4x4 grid flattened
action_size = 4  # left, right, up, down
batch_size = 64

agent = DQNAgent(state_size, action_size, batch_size, DuelingDQN, model_path)
repetition_allowance = 10
video_length = 30  #seconds

# Main loop for multiple training and testing phases
last_average_score = None  # Track the last average score for performance comparison
num_phases = 100000
num_episodes_per_phase = 25
num_actions_per_episode = 5
num_outputs_per_episode = 5
num_tests = 10

save_model_interval = 1000
save_plot_interval = 1
save_video_interval = num_phases // 100
performance_threshold = 10
cross_phase_total_score = 0

for phase in range(num_phases):
    phase += phase_offset
    phase_total_score = 0
    episode_details = []  # To store total score and buffer for each episode

    print(f"============== Starting Training Phase {phase + 1} ==============")
    episode_output = []
    for episode in range(num_episodes_per_phase):
        state = game.getRandomState()
        temp_game = Game(state=state)
        episode_score = 0
        episode_buffer = []  # Temporary buffer to store episode's experiences

        for _ in range(episode):
            action = agent.act(state['board'])
            next_state, reward, done = temp_game.step(action)

            # Instead of pushing directly to the main replay buffer, store in the episode buffer
            episode_buffer.append((state['board'], action, reward, next_state['board'], done))

            state = next_state
            episode_score += reward
            if done:
                break

        # Store this episode's total score and its experiences
        episode_details.append((episode_score, episode_buffer))
        phase_total_score += episode_score

        # Optional: Output episode summaries
        slice_size = (num_episodes_per_phase // num_outputs_per_episode)
        if episode % slice_size == 0 and episode > 0:
            episode_output.append([f"{episode - slice_size}-{episode}",
                                      format(sum([det[0] for det in episode_details[episode - slice_size:episode]]) / slice_size, '.2f'),
                                      agent.epsilon,
                                      agent.optimizer.param_groups[0]['lr']])

    print(tabulate(episode_output, headers=['Episode Range', 'Average Score', 'Epsilon', 'Learning Rate']))

    # After all episodes in a phase are complete, decide which to learn from
    # For simplicity, let's learn from top N% of episodes
    top_n_percent = 10
    episode_details.sort(key=lambda x: x[0], reverse=True)  # Sort episodes by score, descending
    top_episodes = episode_details[:max(1, len(episode_details) * top_n_percent // 100)]

    # Now push experiences from top episodes into the main replay buffer and trigger learning
    for _, buffer in top_episodes:
        for experience in buffer:
            agent.replay_buffer.push(*experience)
        agent.learn()  # Learn from the accumulated experiences of top-performing episodes

    # Continue with model saving and performance plotting as before
    if phase % save_model_interval == 0:
        torch.save(agent.model.state_dict(), f"Models/{curr_dt}/model_phase_{phase + 1}.pth")
    if phase % save_video_interval == 0:
        save_videos(phase)
    print(
        f"End of Training Phase {phase + 1}\t|\tTotal Reward: {format(phase_total_score, '.2f')}\t|\tAverage Reward: {phase_total_score / (phase + 1)}")

    average_score, high_tile, moves_before_break = test_agent(game, agent, num_tests=num_tests)
    with open(f"Outputs/{curr_dt}/average_scores.csv", "a") as f:
        f.write(f"{phase + 1},{average_score}\n")
    with open(f"Outputs/{curr_dt}/high_tiles.csv", "a") as f:
        f.write(f"{phase + 1},{high_tile}\n")
    with open(f"Outputs/{curr_dt}/moves_before_break.csv", "a") as f:
        f.write(f"{phase + 1},{moves_before_break}\n")
    if phase % save_plot_interval == 0:
        save_plot("average_scores")
        save_plot("high_tiles")
        save_plot("moves_before_break")

    print(tabulate([[phase + 1, average_score, high_tile, moves_before_break]], headers=['Phase', 'Average Score', 'High Tile', 'Moves Before Break']))

    if average_score > performance_threshold:
        performance_threshold += (average_score - performance_threshold) * 1.25
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = max(agent.learning_rate_min, param_group[
                'lr'] * agent.learning_rate_decay) if last_average_score is not None and average_score <= last_average_score else \
                param_group['lr']
        # Print color green
        print("\033[30;102m")
        print(f"Performance threshold reached, adjusted performance threshold to {performance_threshold}")
        print("\033[0m")
    else:
        performance_threshold *= 0.99999
        agent.epsilon = min(agent.epsilon_max,
                            agent.epsilon * (1/(agent.epsilon_decay+((1-agent.epsilon_decay)/4))))  # Slightly increase epsilon if performance is below threshold
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = min(agent.learning_rate_max, param_group[
                'lr'] * 1.000001) if last_average_score is not None and average_score <= last_average_score else \
                param_group['lr']
        # Print color red
        print("\033[30;101m")
        print(f"Performance threshold NOT reached, adjusted performance threshold to {performance_threshold}")
        print("\033[0m")

    last_average_score = average_score
    print(
        f"End of Phase {phase + 1}\t|\tAdjusted Epsilon: {agent.epsilon}\t|\tLearning Rate: {agent.optimizer.param_groups[0]['lr']}")
    print("=====================================================================")

print("==============\t|\tFinal Testing\t|\t==============")
torch.save(agent.model.state_dict(), f"Models/{curr_dt}/final_model.pth")
test_agent(game, agent, num_tests=20)
save_videos("final")

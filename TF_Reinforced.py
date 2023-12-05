import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any
import tensorflow as tf
import tensorflow_probability as tfp
import os
import random
from datetime import datetime

# Check TensorFlow version and GPU availability
print('TensorFlow version:', tf.__version__)
print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)

# Your game environment
from Game import Game  # Ensure this is compatible with TensorFlow

class Policy(tf.keras.Model):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = tf.keras.layers.Dense(128, activation='relu')
        self.affine2 = tf.keras.layers.Dense(128, activation='relu')
        self.affine3 = tf.keras.layers.Dense(128, activation='relu')
        self.affine4 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.6)
        self.action_head = tf.keras.layers.Dense(4, activation=None)
        self.saved_log_probs = []  # Added this line

    def call(self, inputs, training=False):
        x = self.affine1(inputs)
        x = self.affine2(x)
        x = self.affine3(x)
        x = self.affine4(x)
        x = self.dropout(x, training=training)
        action_scores = self.action_head(x)
        return tf.nn.softmax(action_scores)

def one_hot_encode_action(action: int, num_actions: int = 4) -> tf.Tensor:
    """Create a one-hot encoded vector for the action."""
    return tf.one_hot(action, num_actions)

def select_action(state, model, epsilon=0.1):
    if random.random() < epsilon:
        return random.randint(0, 3), None  # Random action

    state_tensor = tf.convert_to_tensor(state['board'], dtype=tf.float32)
    action_probs = model(state_tensor, training=False)
    dist = tfp.distributions.Categorical(probs=action_probs)
    action = dist.sample()

    log_prob = dist.log_prob(action)
    return int(action.numpy()), log_prob

def compute_loss(saved_log_probs, discounted_rewards):
    loss = 0
    for log_prob, reward in zip(saved_log_probs, discounted_rewards):
        loss += -log_prob * reward
    return loss

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    running_add = 0
    for reward in reversed(rewards):
        running_add = running_add * gamma + reward
        discounted_rewards.insert(0, running_add)
    return discounted_rewards

@tf.function
def train_step(model, optimizer, rewards, saved_log_probs):
    discounted_rewards = compute_discounted_rewards(rewards)

    with tf.GradientTape() as tape:
        loss = compute_loss(saved_log_probs, discounted_rewards)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(env: Game, policy: Policy, max_steps: int = 1000) -> Tuple[float, int]:
    """Evaluate the policy in the given environment."""
    state = env.reset()
    total_reward = 0.0
    step = 0
    done = False

    while not done and step < max_steps:
        action = select_action(state, policy, epsilon=0)  # Greedy action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        step += 1
        env.render()  # Uncomment if rendering is desired

    return total_reward, step

def finish_episode(policy: Policy, rewards: List[float], gamma: float = 0.99) -> tf.Tensor:
    """Calculates discounted rewards and updates the policy."""
    discounted_rewards = []
    running_add = 0
    for r in reversed(rewards):
        running_add = running_add * gamma + r
        discounted_rewards.insert(0, running_add)

    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
    mean = tf.reduce_mean(discounted_rewards)
    std = tf.math.reduce_std(discounted_rewards) + 1e-9
    returns = (discounted_rewards - mean) / std

    loss = train_step(policy, optimizer, returns)
    policy.saved_log_probs = []
    return loss

def calculate_reward(state: Dict[str, tf.Tensor], next_state: Dict[str, tf.Tensor], move_repeated: bool) -> float:
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
        score_delta = next_state['score']
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
    elif next_state['total_empty_cells'] == state['total_empty_cells'] and move_repeated:
        same_count += 1
        if same_count > 0:
            total_empty_cells_delta = -1
        else:
            total_empty_cells_delta = 0
    else:
        total_empty_cells_delta = 0
        same_count = 0

    # Win/Loss Reward
    if next_state['win'] and not win_flag:
        win_flag = True
        win_value = 0 #10
        loss_value = 0
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
    return reward
env = Game()
policy = Policy()
optimizer = tf.keras.optimizers.Adam()
def main():
    logger = ReinforcementLearningLogger(policy, output_dir="./Outputs/")
    num_episodes = 1000
    test_interval = 100  # Define how often you want to test
    non_repeating_moves = []
    repeating_moves = []

    for episode in range(num_episodes):
        num_repeating_moves= 0
        num_non_repeating_moves = 0
        env.reset()
        state = env.getState()
        episode_reward = 0
        done = False
        last_action = -1  # Initialize last_action, -1 can indicate no previous action

        # Training loop
        while not done:
            action, _ = select_action(state, policy)  # Updated call
            next_state, reward, done, _ = env.step(action)
            if action == last_action:
                num_repeating_moves += 1
            else:
                num_non_repeating_moves += 1
            reward = calculate_reward(state, next_state, action == last_action)
            policy.saved_log_probs.append(reward)
            state = next_state
            last_action = action  # Update last_action
            episode_reward += reward
        non_repeating_moves.append(num_non_repeating_moves)
        repeating_moves.append(num_repeating_moves)

        loss = finish_episode(policy, policy.rewards, optimizer)
        print(f"Episode {episode}: Reward = {episode_reward}, Loss = {loss}")
        policy.rewards = []

        # Periodically test the policy
        if episode % test_interval == 0 or episode == num_episodes - 1:
            total_reward, steps_taken = test_step(env, policy)
            print(f"Test Episode {episode}: Total Reward = {total_reward}, Steps = {steps_taken}")

        # Logging
        if episode_reward > 0 and episode_reward % 100 == 0:
            logger.log_episode(episode, episode_reward, env.getMaxValue() , non_repeating_moves, repeating_moves)

class ReinforcementLearningLogger:
    def __init__(self, policy: Policy, output_dir: str = "./Outputs/"):
        self.policy = policy
        self.output_dir = output_dir
        self.rewards_history = []
        self.max_tiles_history = []
        self.non_repeating_moves_history = []
        self.repeating_moves_history = []

    def log_episode(self, episode: int, reward: float, max_tile: int, non_repeating_moves: int, repeating_moves: int):
        """Logs the results of an episode."""
        self.rewards_history.append(reward)
        self.max_tiles_history.append(max_tile)
        self.non_repeating_moves_history.append(non_repeating_moves)
        self.repeating_moves_history.append(repeating_moves)

        if episode % 50 == 0:
            self.save_policy(f"policy_{datetime.now().strftime('%Y%m%d-%H%M')}.h5")

        if episode % 1000 == 0 and episode > 0:
            self.plot_results(episode)

    def save_policy(self, filename: str):
        """Saves the policy model."""
        self.policy.save(os.path.join(self.output_dir, filename))

    def plot_results(self, episode: int):
        """Plots the historical data of the training process."""
        x = np.arange(len(self.rewards_history))

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        self._plot_with_trendline(x, self.rewards_history, 'Rewards', episode)

        plt.subplot(2, 2, 2)
        self._plot_with_trendline(x, self.max_tiles_history, 'Max Tiles', episode)

        plt.subplot(2, 2, 3)
        self._plot_with_trendline(x, self.non_repeating_moves_history, 'Non-Repeating Moves', episode)

        plt.subplot(2, 2, 4)
        self._plot_with_trendline(x, self.repeating_moves_history, 'Repeating Moves', episode)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'results_{datetime.now().strftime("%Y%m%d-%H%M")}_{episode}.png'))
        plt.close()

    def _plot_with_trendline(self, x: np.ndarray, y: List[float], title: str, episode: int):
        """Plots individual graphs with trendlines."""
        plt.scatter(x, y)
        trend = np.polyfit(x, y, 3)
        plt.plot(x, np.polyval(trend, x), color="red")
        plt.title(f'{title} after episode {episode}')
        plt.xlabel('Episode')
        plt.ylabel(title)


if __name__ == "__main__":
   main()
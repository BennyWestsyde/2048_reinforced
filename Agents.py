from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

from KEYS import device


class ReplayBuffer:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		return random.sample(self.buffer, batch_size)

	def __len__(self):
		return len(self.buffer)


@dataclass
class epsilon:
	max: float = 1.0
	decay: float = 0.995
	min: float = 0.01
	_value: float = field(compare=True, default=1.0)

	def __set__(self, instance, val):
		if val > self.max:
			self._value = self.max
		elif val < self.min:
			self._value = self.min
		else:
			self._value = val

	def __eq__(self, other):
		return self._value == other

	def __lt__(self, other):
		return self._value < other

	def __gt__(self, other):
		return self._value > other

	def __le__(self, other):
		return self._value <= other

	def __ge__(self, other):
		return self._value >= other

	def __ne__(self, other):
		return self._value != other

	def __add__(self, other):
		return self._value + other

	def __sub__(self, other):
		return self._value - other

	def __mul__(self, other):
		return self._value * other

	def __truediv__(self, other):
		return self._value / other

	def __floordiv__(self, other):
		return self._value // other

	def __mod__(self, other):
		return self._value % other

	def __pow__(self, other):
		return self._value ** other

	def __float__(self):
		return float(self._value)


class DQNAgent:
	def __init__(self, state_size, action_size, batch_size, model, model_path=None):
		self.state_size = state_size
		self.action_size = action_size
		self.batch_size = batch_size
		self.epsilon = 0.5
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.epsilon_max = 0.75
		self.learning_rate_decay = 0.999
		self.learning_rate_min = 1e-5
		self.learning_rate_max = 1e-3
		self.gamma = 0.99  # Discount rate
		self.model = model()
		if model_path:
			self.model.load_state_dict(torch.load(model_path))
		self.model.to(device)
		self.optimizer = optim.Adam(self.model.parameters())
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10)
		self.replay_buffer = ReplayBuffer(10000)
		self.losses = []

	def act(self, state, last_reward=0, test=False):
		state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(
			device)  # Ensure tensor is on the correct device
		q_values = self.model(state_tensor).detach()  # No need to send to CPU or convert to numpy
		q_value_range = torch.max(q_values) - torch.min(q_values)  # Calculate range of Q-values directly with PyTorch

		# Dynamically adjust epsilon based on Q-value range
		sensitivity = 0.000001  # Adjust this based on your environment
		if last_reward > 0:
			self.epsilon = max(self.epsilon_min,
			                   min(self.epsilon_max, self.epsilon - sensitivity * q_value_range.item()))
		else:
			self.epsilon = min(self.epsilon_max,
			                   max(self.epsilon_min, self.epsilon + (sensitivity * q_value_range.item()) / 2))
		if test or np.random.rand() > self.epsilon:
			return torch.argmax(q_values).item()  # Use torch.argmax and convert to Python int
		else:
			return np.random.randint(0, self.action_size)

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
		self.losses.append(loss.item())
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()

	def push_replay_buffer(self, state, action, reward, next_state, done):
		"""
		Pushes the arguments into the replay buffer
		:param state:
		:param action:
		:param reward:
		:param next_state:
		:param done:
		:return:
		"""
		self.replay_buffer.push(state, action, reward, next_state, done)

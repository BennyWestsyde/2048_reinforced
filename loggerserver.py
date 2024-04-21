import os
import pickle
import socket
import subprocess
import sys
import time
from io import BytesIO, StringIO
from multiprocessing import freeze_support

import progressbar

from KEYS import num_episodes_per_phase, num_phases, num_tests, percentage_kept, slice_size
from dataStructures import EpochTestingDetails, EpochTrainingDetails, PhaseDetails, PhaseBuffer, \
	TestingOutput, \
	TrainingOutput


class LoggerServer:
	def __init__(self, pipe):
		self.pipe = pipe
		self.phase_details = PhaseDetails(phase_total_score=0, phase_high_tile=0, phase_moves_before_break=0,
		                                  phase_game_break=0, phase_training_details=None, phase_testing_details=None)
		self.phase_buffer = PhaseBuffer(phase_average_score=0,
		                                phase_average_high_tile=0,
		                                phase_average_moves_before_break=0,
		                                phase_average_game_break=0,
		                                phase_testing_output=None,
		                                phase_training_output=None)
		self.phase_details_list = []
		self.training_episodes = []
		self.training_output = []
		self.testing_episodes = []
		self.testing_output = []
		self.fd = None

		self.progress = {
			"overall": 0,
			"phase_training": 0,
			"training_episodes": [0 for _ in range(num_episodes_per_phase)],
			"phase_testing": 0,
			"testing_episodes": [0 for _ in range(num_tests)]
			}

	def append_training_episode(self, episode_details):
		self.training_episodes.append(episode_details)

	def append_testing_episode(self, episode_details):
		self.testing_episodes.append(episode_details)

	def append_training_output(self, game, learning_rate, epsilon):
		self.training_output.append(TrainingOutput(
			epoch_range=f"{game - slice_size}-{game}",
			epoch_average_score=sum([ed.episode_score for ed in self.training_episodes[-slice_size:]]) / slice_size,
			epsilon=epsilon,
			learning_rate=learning_rate))

	def append_testing_output(self, game):
		self.testing_output.append(TestingOutput(
			game=game,
			score=self.testing_episodes[-1].score,
			high_tile=self.testing_episodes[-1].high_tile,
			moves_before_break=self.testing_episodes[-1].moves_before_break,
			game_output=self.testing_episodes[-1].game_output))

	def append_phase_training_details(self):
		self.phase_details.phase_training_details = EpochTrainingDetails(
			epoch_training_score=sum([episode.score for episode in self.training_episodes]),
			epoch_training_high_tile=max([episode.high_tile for episode in self.training_episodes]),
			epoch_training_moves_before_break=sum([episode.moves_before_break for episode in self.training_episodes]),
			epoch_training_game_break=sum([episode.game_break for episode in self.training_episodes]),
			episode_details=self.training_episodes)
		self.progress_bars["phase_training"].variables = {
			"average_score": sum([ed.episode_score for ed in self.training_episodes]) / num_episodes_per_phase,
			"epsilon": self.training_output[-1].epsilon,
			"learning_rate": self.training_output[-1].learning_rate
			}
		self.progress_bars["phase_training"].widgets = [
			'Phase {}: '.format(len(self.phase_details_list) + 1),
			' ',
			progressbar.Bar(marker=progressbar.GranularBar(marker='█'), left='|', right='|', fill='█'),
			' | ',
			progressbar.Variable('average_score'),
			' | ',
			progressbar.Variable('epsilon'),
			' | ',
			progressbar.Variable('learning_rate')
			]
		self.progress_bars["phase_training"].update(num_episodes_per_phase)
		progressbar.streams.flush()
		self.progress_bars["phase_training"].finish()

	def append_phase_testing_details(self):
		self.phase_details.phase_testing_details = EpochTestingDetails(
			epoch_testing_score=sum([episode.score for episode in self.testing_episodes]),
			epoch_testing_high_tile=max([episode.high_tile for episode in self.testing_episodes]),
			epoch_testing_moves_before_break=sum([episode.moves_before_break for episode in self.testing_episodes]),
			epoch_testing_game_break=sum([episode.game_break for episode in self.testing_episodes]),
			epoch_testing_game_loss=sum([episode.game_loss for episode in self.testing_episodes]),
			epoch_testing_game_win=sum([episode.game_win for episode in self.testing_episodes]),
			episode_details=self.testing_episodes)
		self.progress_bars["phase_testing"].variables = {
			"average_score": sum([ed.score for ed in self.testing_episodes]) / num_tests,
			"average_high_tile": sum([ed.high_tile for ed in self.testing_episodes]) / num_tests,
			"average_moves_before_break": sum([ed.moves_before_break for ed in self.testing_episodes]) / num_tests,
			"average_game_break": sum([ed.game_break for ed in self.testing_episodes]) / num_tests
			}
		self.progress_bars["phase_testing"].widgets = [
			'Phase {}: '.format(len(self.phase_details_list) + 1),
			' ',
			progressbar.Bar(marker=progressbar.GranularBar(marker='█'), left='|', right='|', fill='█'),
			' | ',
			progressbar.Variable('average_score'),
			' | ',
			progressbar.Variable('average_high_tile'),
			' | ',
			progressbar.Variable('average_moves_before_break'),
			' | ',
			progressbar.Variable('average_game_break')
			]
		self.progress_bars["phase_testing"].update(num_tests)
		progressbar.streams.flush()
		self.progress_bars["phase_testing"].finish()

	def append_phase_details(self):
		self.phase_details.phase_total_score = self.phase_details.phase_training_details.epoch_training_score + self.phase_details.phase_testing_details.epoch_testing_score
		self.phase_details.phase_high_tile = self.phase_details.phase_testing_details.epoch_testing_high_tile
		self.phase_details.phase_moves_before_break = self.phase_details.phase_testing_details.epoch_testing_moves_before_break
		self.phase_details.phase_game_break = self.phase_details.phase_testing_details.epoch_testing_game_break
		self.phase_details_list.append(self.phase_details)
		self.phase_details = PhaseDetails(phase_total_score=0, phase_high_tile=0, phase_moves_before_break=0,
		                                  phase_game_break=0, phase_training_details=None, phase_testing_details=None)
		self.training_episodes = []
		self.testing_episodes = []
		self.training_output = []
		self.testing_output = []

	def append_phase_buffer(self):
		self.phase_buffer.phase_average_score = sum(
			[phase.phase_total_score for phase in self.phase_details_list]) / len(self.phase_details_list)
		self.phase_buffer.phase_average_high_tile = sum(
			[phase.phase_high_tile for phase in self.phase_details_list]) / len(self.phase_details_list)
		self.phase_buffer.phase_average_moves_before_break = sum(
			[phase.phase_moves_before_break for phase in self.phase_details_list]) / len(self.phase_details_list)
		self.phase_buffer.phase_average_game_break = sum(
			[phase.phase_game_break for phase in self.phase_details_list]) / len(self.phase_details_list)

	def update_episode(self, episode_number, step_number):
		self.progress_bars["training_episodes"][episode_number].update(step_number)

	def update_game(self, game_number, step_number):
		self.progress_bars["testing_episodes"][game_number].update(step_number)

	def update_training(self, step_number):
		self.progress_bars["phase_training"].update(step_number)

	def update_testing(self, step_number):
		self.progress_bars["phase_testing"].update(step_number)

	def update_overall(self, step_number):
		self.progress_bars["overall"].update(step_number)

	def start(self):
		if self.progress_bars["overall"].start_time is None:
			self.progress_bars["overall"].start()
			self.progress_bars["overall"].update(0)
		if self.progress_bars["phase_training"].start_time is None:
			self.progress_bars["phase_training"].start()
			self.progress_bars["phase_training"].update(0)
		if self.progress_bars["phase_testing"].start_time is None:
			self.progress_bars["phase_testing"].start()
			self.progress_bars["phase_testing"].update(0)
		for bar in self.progress_bars["training_episodes"]:
			if bar.start_time is None:
				bar.start()
				bar.update(0)
		for bar in self.progress_bars["testing_episodes"]:
			if bar.start_time is None:
				bar.start()
				bar.update(0)
		progressbar.streams.flush()

	def get_top_episodes(self):
		top_episodes = self.training_episodes[:max(1, len(self.training_episodes) * percentage_kept // 100)]
		self.send_data(top_episodes)

	def receive_data(self):
		output = os.read(self.pipe.fileno(), 4096)
		if output is None:
			return None
		return pickle.loads(output)

	def send_data(self, data):
		serialized_data = pickle.dumps(data)
		os.write(self.pipe.fileno(), serialized_data)

	def mainloop(self):
		self.fd = progressbar.LineOffsetStreamWrapper(lines=os.get_terminal_size().lines, stream=sys.stdout)
		self.progress_bars = {
			"overall": progressbar.ProgressBar(max_value=num_phases, redirect_stdout=True, fd=self.fd, line_offset=0),
			"phase_training": progressbar.ProgressBar(max_value=num_episodes_per_phase, redirect_stdout=True,
			                                          fd=self.fd, line_offset=2),
			"training_episodes": [progressbar.ProgressBar(max_value=slice_size,
			                                              redirect_stdout=True,
			                                              fd=self.fd,
			                                              line_offset=(3 + (i % (num_episodes_per_phase // 2)))) for i
			                      in range(num_episodes_per_phase)],
			"phase_testing": progressbar.ProgressBar(max_value=num_tests,
			                                         redirect_stdout=True,
			                                         fd=self.fd,
			                                         line_offset=(3 + (num_episodes_per_phase // 2) + 1)),
			"testing_episodes": [progressbar.ProgressBar(max_value=slice_size,
			                                             redirect_stdout=True,
			                                             fd=self.fd,
			                                             line_offset=(4 + (i % (num_tests)) + 1)) for i in
			                     range(num_tests)]
			}
		while True:
			self.start()
			action = self.receive_data()
			if action is None:
				continue
			input_type = action[0]
			if input_type == 'training_episode':
				self.append_training_episode(action[1])
			elif input_type == 'testing_episode':
				self.append_testing_episode(action[1])
			elif input_type == 'training_output':
				self.append_training_output(action[1], action[2], action[3])
			elif input_type == 'testing_output':
				self.append_testing_output(action[1])
			elif input_type == 'training_details':
				self.append_phase_training_details()
			elif input_type == 'testing_details':
				self.append_phase_testing_details()
			elif input_type == 'phase_details':
				self.append_phase_details()
			elif input_type == 'phase_buffer':
				self.append_phase_buffer()
			elif input_type == 'update':
				if action[1] == 'training_episodes':
					self.update_episode(action[2], action[3])
				elif action[1] == 'testing_episodes':
					self.update_game(action[2], action[3])
				elif action[1] == 'phase_training':
					self.update_training(action[2])
				elif action[1] == 'phase_testing':
					self.update_testing(action[2])
				elif action[1] == 'overall':
					self.update_overall(action[2])
			elif action[1] == 'top_episodes':
				self.get_top_episodes()
			else:
				raise ValueError(f"Invalid input type: {input_type}")


def runServer(port):
	LoggerServer(port).mainloop()


if __name__ == '__main__':
	freeze_support()
	args = sys.argv[1:]
	if len(args) > 0:
		LoggerServer(BytesIO(args[0].encode())).mainloop()
	else:
		LoggerServer().mainloop()

import os
import pickle
import socket
import sys
import time
from multiprocessing import freeze_support


class LoggerInterface:
    def __init__(self, pipe):
        self.pipe = pipe

    def receive_data(self):
        output = os.read(self.pipe.fileno(), 4096)
        if output is None:
            return None
        return pickle.loads(output)

    def send_data(self, data):
        serialized_data = pickle.dumps(data)
        os.write(self.pipe.fileno(), serialized_data)

    def get_top_episodes(self):
        self.send_data(("top_episodes"))
        return self.receive_data()


    def append_training_episode(self, episode_details):
        self.send_data(('training_episode', episode_details))

    def append_testing_episode(self, episode_details):
        self.send_data(('testing_episode', episode_details))

    def append_training_output(self, episode, epsilon, learning_rate):
        self.send_data(("training_output", episode, epsilon, learning_rate))

    def append_testing_output(self, game):
        self.send_data(('testing_output', game))

    def append_phase_training_details(self):
        self.send_data('training_details')

    def append_phase_testing_details(self):
        self.send_data('testing_details')

    def update_episode(self, episode_number, step_number):
        self.send_data(('update', 'training_episodes', episode_number, step_number))

    def update_game(self, game_number, step_number):
        self.send_data(('update', 'testing_episodes', game_number, step_number))

    def update_training(self, step_number):
        self.send_data(('update', 'phase_training', step_number))

    def update_testing(self, step_number):
        self.send_data(('update', 'phase_testing', step_number))

    def update_overall(self, step_number):
        self.send_data(('update', 'overall', step_number))


if __name__ == '__main__':
    freeze_support()
    args = sys.argv[1:]
    if len(args) == 1:
        pipe = args[0]
    else:
        pipe = sys.stdin
    logger = LoggerInterface(pipe)
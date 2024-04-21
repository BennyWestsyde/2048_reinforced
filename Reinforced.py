import multiprocessing
import random
import subprocess
import sys
from time import gmtime, strftime

import numpy
import progressbar as pb
from tabulate import tabulate

import loggerserver
from Agents import DQNAgent
from Game import Game
from KEYS import action_size, batch_size, device, num_actions_per_episode, num_episodes_per_phase, num_phases, \
	num_tests, percentage_kept, repetition_allowance, save_model_interval, save_plot_interval, save_video_interval, \
	slice_size, state_size
from Models import DuelingDQN
from MultiMediaOutput import save_plot, save_video
from ansicolor import color
from dataStructures import Episode, EpisodeDetails, EpochTestingDetails, EpochTrainingDetails, GameState, PhaseBuffer, \
	PhaseDetails, \
	TestingOutput, \
	TrainingOutput
from loggerserver import LoggerServer
from loggerClient import LoggerInterface
from utils import *

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


def train_agent(agent, bar: pb.ProgressBar):
	state = Game().getRandomState()
	temp_game = Game(state=state)
	last_reward = 0
	episode_score = 0
	episode = Episode()

	for _ in range(num_actions_per_episode):
		action = agent.act(state['board'], last_reward=last_reward)
		action_result = temp_game.step(action)
		next_state = action_result[0]
		done = action_result[-1]
		reward = action_result[1]

		game_state = GameState(action=action, state=next_state, reward=reward, done=done)
		episode.states.append(game_state)

		episode_score += reward
		last_reward = reward
		pb.streams.flush()
		if done:
			break
		state = next_state
	return EpisodeDetails(episode_score=episode_score, episode_buffer=episode)


def test_agent(agent, bar: pb.ProgressBar):
	temp_game = Game()
	state = temp_game.getState()
	done = False
	count = 0
	break_flag = False
	prev_action = None
	episode_score = 0
	episode = Episode()
	while not done:
		with torch.no_grad():
			action = agent.act(state['board'], test=True)
			if prev_action == action:
				count += 1
				if count > repetition_allowance:
					break_flag = True
					break
				else:
					bar.update(count)
					pb.streams.flush()
			else:
				count = 0
			prev_action = action

		action_result = temp_game.step(action)
		next_state = action_result[0]
		done = action_result[-1]
		reward = action_result[1]

		game_state = GameState(action=action, state=next_state, reward=reward, done=done)
		episode.states.append(game_state)

		episode_score += reward
		state = next_state

	if break_flag:
		game_outcome = 2
	elif game.checkWin():
		game_outcome = 1
	else:
		game_outcome = 0

	return EpisodeDetails(episode_score=episode_score, episode_buffer=episode)


def teach_agent(agent, top_episodes):
	# Now push experiences from top episodes into the main replay buffer and trigger learning
	loss = []
	for ep in top_episodes:
		for i, game_state in enumerate(ep.episode_buffer.states):
			next_state = ep.episode_buffer.states[i + 1].state if i + 1 < len(
				ep.episode_buffer.states) else game_state.state
			agent.push_replay_buffer(state=game_state.state['board'], action=game_state.action,
			                         reward=game_state.reward, next_state=next_state['board'],
			                         done=game_state.done)
		push_loss = agent.learn()
		if push_loss is not None:
			loss.append(push_loss)
	if len(loss) == 0:
		return 0
	return sum(loss) / len(loss)


def save_videos(curr_phase):
	if curr_phase <= 100:
		return
	p1 = multiprocessing.Process(target=save_video, args=("average_scores", curr_phase))
	p2 = multiprocessing.Process(target=save_video, args=("high_tiles", curr_phase))
	p3 = multiprocessing.Process(target=save_video, args=("moves_before_break", curr_phase))
	p1.start()
	p2.start()
	p3.start()
	p1.join()
	p2.join()
	p3.join()


def load_model(model_path_, model_class):
	temp_model = model_class()  # Instantiate the model
	temp_model.load_state_dict(torch.load(model_path_, map_location=device))  # Load the state dict
	temp_model.to(device)  # Ensure model is on the correct device
	return temp_model


# Initialize game and agent
game = Game()

agent = DQNAgent(state_size, action_size, batch_size, DuelingDQN, model_path)

# Main loop for multiple training and testing phases
last_average_score = None  # Track the last average score for performance comparison

init_filesystem(curr_dt)
#multiprocessing.Queue()
#subprocess.Popen(["python", "loggerserver.py"], stdin=pipe[0], stdout=sys.stdout, stderr=sys.stderr)
#subprocess.Popen(["python","loggerserver.py",f"{port}"], stderr=sys.stderr, stdout=sys.stdout)
#loggerInterface = LoggerInterface(pipe=pipe)
print_fd = pb.LineOffsetStreamWrapper(lines=os.get_terminal_size()[-1], stream=sys.stdout)
print(f"Terminal Size: {os.get_terminal_size()[0]} W x {os.get_terminal_size()[1]} H", file=print_fd)
pb.streams.flush()
phase_bar_offset = 1
training_title = 1
phase_training_bar_offset = phase_bar_offset + 1
training_bar_num = num_episodes_per_phase // 2
testing_title = 1
phase_testing_bar_offset = phase_training_bar_offset + training_bar_num + testing_title + 1
phase_testing_bar_num = num_tests

phase_bar = pb.ProgressBar(
	max_value=num_phases,
	prefix=print_fd.DOWN * phase_bar_offset,
	widgets=[
		pb.Percentage(),
		' ',
		pb.Bar(marker=pb.GranularBar(marker='█'), left='|', right='|', fill='█'),
		' ',
		pb.AdaptiveETA()
		],
	fd=print_fd)
os.system('clear')
phase_bar.start()
for phase in range(num_phases):
	pb.streams.flush()
	phase += phase_offset
	phase_total_score = 0
	phase_details = PhaseDetails(phase_total_score=0, phase_high_tile=0, phase_moves_before_break=0,
	                             phase_game_break=0, phase_training_details=None, phase_testing_details=None)

	#print(f"============== Starting Training Phase {phase + 1} ==============", file=print_fd)
	# Create main progress bar for the phase
	phase_training_bar = pb.ProgressBar(
		max_value=num_episodes_per_phase,
		prefix=print_fd.DOWN * phase_training_bar_offset + 'TRAINING'.center(os.get_terminal_size()[0], "="),
		widgets=[
			'\nPhase {}: '.format(phase + 1),
			pb.Percentage(),
			' ',
			pb.Bar(marker=pb.GranularBar(marker='█'), left='|', right='|', fill='█'),
			' ',
			pb.AdaptiveETA()
			],
		fd=print_fd)
	#
	phase_training_bar.start()
	episode = 0
	training_episodes = []
	training_output = []
	for episode in range(num_episodes_per_phase):
		line_offset = phase_training_bar_offset + training_title + (episode % training_bar_num) + 1
		episode_bar = pb.ProgressBar(
			max_value=num_actions_per_episode,
			prefix=print_fd.DOWN * line_offset,
			widgets=[
				'Episode {}: '.format(episode + 1),
				pb.Percentage(),
				' ',
				pb.Bar(marker=pb.GranularBar(marker='█'), left='|', right='|', fill='█'),
				' ',
				pb.AdaptiveETA()
				],
			fd=print_fd)
		episode_bar.start()
		pb.streams.flush()
		episode_details = train_agent(agent, episode_bar)
		episode_bar.finish()
		#loggerInterface.append_training_episode(episode_details)
		training_episodes.append(episode_details)

		if episode % slice_size == 0 and episode > 0:
			training_output.append(TrainingOutput(
				epoch_range=f"{episode - slice_size + 1}-{episode}",
				epoch_average_score=sum([ed.episode_score for ed in training_episodes[-slice_size:]]) / slice_size,
				epsilon=agent.epsilon,
				learning_rate=agent.optimizer.param_groups[0]['lr']
				))
		#loggerInterface.append_training_output(episode, agent.epsilon, agent.optimizer.param_groups[0]['lr'],)

		phase_training_bar.increment()
		pb.streams.flush()

	#loggerInterface.append_phase_training_details()
	epoch_training_details = EpochTrainingDetails(
		epoch_training_score=sum([ed.episode_score for ed in training_episodes]),
		epoch_training_high_tile=max(
			[max([state.state['max_value'] for state in ed.episode_buffer.states]) for ed in training_episodes]),
		epoch_training_moves_before_break=max([len(ed.episode_buffer.states) for ed in training_episodes]),
		epoch_training_game_break=sum([1 for ed in training_episodes if ed.episode_buffer.states[-1].done]),
		episode_details=training_episodes
		)
	phase_details.phase_training_details = epoch_training_details
	phase_training_bar.finish()

	results_training_bar = pb.ProgressBar(widgets=[
		'\nPhase {}: '.format(phase + 1),
		' ',
		pb.Bar(marker=pb.GranularBar(marker='█'), left='|', right='|', fill='█'),
		' | ',
		pb.Variable('average_score'),
		' | ',
		pb.Variable('epsilon'),
		' | ',
		pb.Variable('learning_rate')
		],
		fd=print_fd,
		variables={
			"average_score": sum([ed.episode_score for ed in training_episodes]) / num_episodes_per_phase,
			"epsilon": agent.epsilon,
			"learning_rate": agent.optimizer.param_groups[0]['lr']
			},
		prefix=print_fd.DOWN * (phase_training_bar_offset + 1)
		)
	phase_training_bar.start(init=True)
	phase_training_bar.update(num_episodes_per_phase)

	#print("=====================================================================", file=print_fd)
	#print(tabulate(training_output,
	#               headers=['Episode Range', 'Average Score', 'Epsilon', 'Learning Rate']))

	# Continue with model saving and performance plotting as before
	if phase % save_model_interval == 0:
		torch.save(agent.model.state_dict(), f"Models/{curr_dt}/model_phase_{phase + 1}.pth")
	if phase % save_video_interval == 0:
		save_videos(phase)

	#print(
	#   f"End of Training Phase {phase + 1}\t|\tTotal Reward: {format(phase_total_score, '.2f')}\t|\tAverage Reward: {phase_total_score / (phase + 1)}")
	#print(f"============== Starting Testing Phase {phase + 1} ==============", file=print_fd)
	testing_episodes = []
	testing_output = []
	phase_test_bar = pb.ProgressBar(
		max_value=num_tests,
		prefix=print_fd.DOWN * phase_testing_bar_offset + 'TESTING'.center(os.get_terminal_size()[0], "="),
		widgets=[
			'\nPhase {}: '.format(phase + 1),
			pb.Percentage(),
			' ',
			pb.Bar(marker=pb.GranularBar(marker='█'), left='|', right='|', fill='█'),
			' ',
			pb.AdaptiveETA()
			],
		redirect_stdout=True,
		fd=print_fd)
	phase_test_bar.start()

	for i in range(num_tests):
		bar_offset = phase_testing_bar_offset + testing_title + (i % phase_testing_bar_num) + 1
		testing_bar = pb.ProgressBar(
			max_value=repetition_allowance + 1,
			prefix=print_fd.DOWN * bar_offset,
			widgets=[
				'Game {}: '.format(i + 1),
				pb.Percentage(),
				' ',
				pb.Bar(marker=pb.GranularBar(marker='█'), left='|', right='|', fill='█'),
				' ',
				pb.AdaptiveETA()
				],
			fd=print_fd)
		testing_bar.start()
		game_output = test_agent(agent, testing_bar)

		#loggerInterface.append_testing_episode(game_output)

		#loggerInterface.append_testing_output(i)
		#
		testing_episodes.append(game_output)
		testing_output.append(TestingOutput(
			game=i + 1,
			score=testing_episodes[-1].episode_score,
			high_tile=max([state.state['max_value'] for state in testing_episodes[-1].episode_buffer.states]),
			moves_before_break=len(testing_episodes[-1].episode_buffer.states),
			game_output=1 if testing_episodes[-1].episode_buffer.states[-1].done else 0
			))
		testing_bar.finish()
		#testing_output.append(game_output)
		phase_test_bar.increment()
		pb.streams.flush()
	phase_test_bar.finish()

	#loggerInterface.append_phase_testing_details()

	epoch_testing_details = EpochTestingDetails(
		epoch_testing_score=sum([ed.episode_score for ed in testing_episodes]),
		epoch_testing_high_tile=max(
			[max([state.state['max_value'] for state in ed.episode_buffer.states]) for ed in testing_episodes]),
		epoch_testing_moves_before_break=max([len(ed.episode_buffer.states) for ed in testing_episodes]),
		epoch_testing_game_break=sum([1 for ed in testing_episodes if ed.episode_buffer.states[-1].done]),
		epoch_testing_game_loss=sum([1 for ed in testing_episodes if not ed.episode_buffer.states[-1].done]),
		epoch_testing_game_win=sum([1 for ed in testing_episodes if ed.episode_buffer.states[-1].done]),
		episode_details=testing_episodes
		)

	phase_details.phase_total_score = sum([ed.episode_score for ed in testing_episodes])
	phase_details.phase_high_tile = max(
		[max([state.state['max_value'] for state in ed.episode_buffer.states]) for ed in testing_episodes])
	phase_details.phase_moves_before_break = max([len(ed.episode_buffer.states) for ed in testing_episodes])
	phase_details.phase_game_break = sum([1 for ed in testing_episodes if ed.episode_buffer.states[-1].done])
	phase_details.phase_testing_details = epoch_testing_details
	phase_details.phase_training_details = epoch_training_details

	total_scores = [ed.episode_score for ed in testing_episodes]
	total_high_tile = [max([state.state['max_value'] for state in ed.episode_buffer.states]) for ed in testing_episodes]
	total_moves_before_break = [len(ed.episode_buffer.states) for ed in testing_episodes]
	game_outputs = [1 if ed.episode_buffer.states[-1].done else 0 for ed in testing_episodes]
	break_count = sum([1 for ed in testing_episodes if ed.episode_buffer.states[-1].done])

	phase_test_bar.variables = {
		"high_score": max(total_scores),
		"highest_tile": max(total_high_tile),
		"most_moves_before_break": max(total_moves_before_break),
		"wins": game_outputs.count(1),
		"losses": game_outputs.count(0),
		"breaks": break_count
		}
	phase_test_bar.finish()

	results_testing_bar = pb.ProgressBar(widgets=[
		'\nPhase {}: '.format(phase + 1),
		' ',
		pb.Bar(marker=pb.GranularBar(marker='█'), left='|', right='|', fill='█'),
		' | ',
		pb.Variable('high_score'),
		' | ',
		pb.Variable('highest_tile'),
		' | ',
		pb.Variable('most_moves_before_break'),
		' | ',
		pb.Variable('wins'),
		' | ',
		pb.Variable('losses'),
		' | ',
		pb.Variable('breaks')
		],
		fd=print_fd,
		variables={
			"high_score": max(total_scores),
			"highest_tile": max(total_high_tile),
			"most_moves_before_break": max(total_moves_before_break),
			"wins": game_outputs.count(1),
			"losses": game_outputs.count(0),
			"breaks": break_count
			},
		prefix=print_fd.DOWN * (phase_testing_bar_offset)
		)

	results_testing_bar.start(init=True)
	results_testing_bar.update(num_tests)

	#Prepare data for tabulation
	#for i, test_output in enumerate(testing_episodes):
	# testing_output.append(TestingOutput(game=i + 1, score=test_output.score, high_tile=test_output.high_tile, moves_before_break=test_output.moves_before_break,
	#                                game_output=color("Win", fg="black", bg="green") if test_output.game_output == 1 else
	#                                color("Loss", fg="black", bg="red") if test_output.game_output == 0 else
	#                                color("Break", fg="black", bg="yellow")))

	#print("=====================================================================", file=print_fd)
	#Print test results using tabulate
	#print(tabulate(testing_output, headers=['Game', 'Score', 'High Tile', 'Moves', 'Game Output']), file=print_fd)
	average_score = numpy.average(total_scores)
	average_break_count = numpy.average(break_count) * 100
	average_high_tile = sum(total_high_tile) / num_tests
	average_moves_before_break = sum(total_moves_before_break) / num_tests
	average_points_per_move = average_score / average_moves_before_break

	#
	with open(f"Outputs/{curr_dt}/average_scores.csv", "a") as f:
		f.write(f"{phase + 1},{average_score}\n")
	with open(f"Outputs/{curr_dt}/high_tiles.csv", "a") as f:
		f.write(f"{phase + 1},{average_high_tile}\n")
	with open(f"Outputs/{curr_dt}/moves_before_break.csv", "a") as f:
		f.write(f"{phase + 1},{average_moves_before_break}\n")
	if phase % save_plot_interval == 0:
		save_plot(curr_dt, "average_scores", phase)
		save_plot(curr_dt, "high_tiles", phase)
		save_plot(curr_dt, "moves_before_break", phase)
	#After all episodes in a phase are complete, decide which to learn from
	#For simplicity, let's learn from top N% of episodes
	#episode_details.sort(key=lambda x: x[0], reverse=True)  Sort episodes by score, descending
	top_episodes = sorted(testing_episodes, key=lambda x: x.episode_score, reverse=True)[
	               :int(len(testing_episodes) * (percentage_kept / 100))]
	#top_episodes = loggerInterface.get_top_episodes()
	loss = teach_agent(agent, top_episodes)
	agent.scheduler.step(-1 * average_moves_before_break)

	#print(tabulate([[phase + 1, average_score, average_high_tile, average_moves_before_break,
	#                 loss,
	#                 repetition_allowance, performance_threshold]],
	#               headers=['Phase', 'Average Score', 'High Tile', 'Moves Before Break', 'Loss', 'Repetition Allowance',
	#                        'Performance Threshold']))

	#if average_score > performance_threshold:
	#    repetition_allowance = max(1, repetition_allowance - 1)
	#     #performance_threshold += (average_score - performance_threshold)  #* 1.25
	#     #agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
	#     Print color green
	#     #color(f"Performance threshold reached, adjusted performance threshold to {performance_threshold}", "black",
	#          "green")
	# #else:
	#     #performance_threshold *= 0.9999
	#     #agent.epsilon = min(agent.epsilon_max,
	#                        agent.epsilon * (1 / (agent.epsilon_decay + ((1 - agent.epsilon_decay) / 4))))  Slightly increase epsilon if performance is below threshold
	#     Print color red
	#     #color("Performance threshold NOT reached, adjusted performance threshold to {performance_threshold}", "black",
	#          "red")

	last_average_score = average_score
	#print(
	#    f"End of Phase {phase + 1}\t|\tAdjusted Epsilon: {agent.epsilon}\t|\tLearning Rate: {agent.optimizer.param_groups[0]['lr']}")
	#print("=====================================================================", file=print_fd)
	phase_bar.increment()
	pb.streams.flush()

print("==============\t|\tFinal Testing\t|\t==============", file=print_fd)
torch.save(agent.model.state_dict(), f"Models/{curr_dt}/final_model.pth")
final_testing_details = []
for i in range(num_tests):
	game_output = test_agent(agent, phase_test_bar)
	final_testing_details.append(TestingOutput(
		game=i + 1,
		score=game_output.episode_score,
		high_tile=max([state.state['max_value'] for state in game_output.episode_buffer.states]),
		moves_before_break=len(game_output.episode_buffer.states),
		game_output=1 if game_output.episode_buffer.states[-1].done else 0
		))
	#loggerInterface.append_testing_output(i)
	#loggerInterface.append_testing_episode(game_output)
	#final_testing_details.append(game_output)
	#phase_test_bar.increment()
	pb.streams.flush()

final_total_scores = [det[0] for det in final_testing_details]
final_high_tile = [max([max(row) for row in det[1]]) for det in final_testing_details]
final_moves_before_break = [len(det[1]) for det in final_testing_details]
final_game_outputs = [det[2] for det in final_testing_details]
final_break_count = sum([1 for det in final_game_outputs if det == 2])
print(tabulate(final_testing_details, headers=['Game', 'Score', 'High Tile', 'Moves', 'Game Output']), file=print_fd)
average_score = numpy.average(final_total_scores)
average_break_count = numpy.average(final_break_count) * 100
average_high_tile = sum(final_high_tile) / num_tests
average_moves_before_break = sum(final_moves_before_break) / num_tests
average_points_per_move = average_score / average_moves_before_break
with open(f"Outputs/{curr_dt}/average_scores.csv", "a") as f:
	f.write(f"Final,{average_score}\n")
with open(f"Outputs/{curr_dt}/high_tiles.csv", "a") as f:
	f.write(f"Final,{average_high_tile}\n")
with open(f"Outputs/{curr_dt}/moves_before_break.csv", "a") as f:
	f.write(f"Final,{average_moves_before_break}\n")
save_videos("final")

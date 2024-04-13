import multiprocessing
import sys
from time import gmtime, strftime

import numpy
from tabulate import tabulate

import KEYS
from Agents import DQNAgent
from Game import Game
from KEYS import repetition_allowance, num_phases, num_episodes_per_phase, num_actions_per_episode, num_tests, \
	save_model_interval, save_plot_interval, save_video_interval, performance_threshold, slice_size, \
	percentage_kept, cpu_cores, device, state_size, action_size, batch_size, num_actions_per_episode
from Models import DuelingDQN
from MultiMediaOutput import save_plot, save_video
from ansicolor import color
from utils import *
import progressbar as pb

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
	episode_details = ()  # episode_details = (score, (episode_buffer,))
	episode_buffer = ()  # episode_buffer = (episode_state1, episode_state2, ...)
	episode_state = ()  # episode_state = (action, state, reward, done)

	# episode_buffer = (action, state, reward, done)
	for _ in range(num_actions_per_episode):
		action = agent.act(state['board'], last_reward=last_reward)
		episode_state = (action,) + temp_game.step(action)
		next_state = episode_state[1]
		episode_score += episode_state[-2]
		last_reward = episode_state[-2]
		bar.increment()
		if episode_state[-1]:
			break
		# Store this episode's total score and its experiences
		state = next_state
		episode_buffer += (episode_state,)
	return (episode_score, episode_buffer,)


def test_agent(agent, bar: pb.ProgressBar):
	test_buffer = ()  # test_buffer = (action, state, reward, done)
	state = Game().getState()
	done = False
	count = 0
	break_flag = False
	prev_action = None
	episode_score = 0
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
		test_buffer = (action,) + game.step(action)
		next_state = test_buffer[1]
		done = test_buffer[-1]
		reward = test_buffer[-2]
		episode_score += reward
		bar.increment()
		state = next_state
	if break_flag:
		game_outcome = 2
	elif game.checkWin():
		game_outcome = 1
	else:
		game_outcome = 0

	return ((episode_score, game_outcome, test_buffer,),)


def teach_agent(agent, top_episodes):
	# Now push experiences from top episodes into the main replay buffer and trigger learning
	for _, buffer in top_episodes:
		for experience in buffer:
			agent.push_replay_buffer(*experience)
		agent.learn()  # Learn from the accumulated experiences of top-performing episodes


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

# Create a file descriptor for regular printing as well

phase_buffer = ()  # phase_buffer = (phase_total_score, high_tile, moves_before_break, game_output,(training_details), (testing_details))

init_filesystem(curr_dt)
for phase in range(num_phases):
	phase += phase_offset
	phase_total_score = 0
	training_details = ()  # training_details = (epoch_training_score, (episode_buffer,))
	training_output = ()  # training_output = (episode_range, average_score, epsilon, learning_rate)
	testing_details = ()  # testing_details = (epoch_testing_score, game_output, (action, state, reward, done))
	testing_output = ()  # testing_output = (game, score, high_tile, moves_before_break, game_output)

	print(f"============== Starting Training Phase {phase + 1} ==============")
	bars = []
	for i in range(num_episodes_per_phase):
		bars.append(
			pb.ProgressBar(
				max_value=num_actions_per_episode,
				# We add 1 to the line offset to account for the `print_fd`
				line_offset=i + 1,
				max_error=False,
				)
			)
	print_fd = pb.LineOffsetStreamWrapper(lines=0, stream=sys.stdout)
	episode = 0
	training_buffer = ()  # training_buffer = (epoch_total_score, (episode_buffer,))
	epoch_buffer = ()  # epoch_buffer = (epoch_
	for episode in range(num_episodes_per_phase):
		training_buffer += train_agent(agent, bars[episode])

		if episode % slice_size == 0 and episode > 0:
			training_output += ([f"{episode - slice_size}-{episode}",
			                     sum([det[0] for det in training_buffer[-slice_size:]]) / slice_size,
			                     agent.epsilon, agent.optimizer.param_groups[0]['lr']],)
	print("=====================================================================")
	print(tabulate(training_output, headers=['Episode Range', 'Average Score', 'Epsilon', 'Learning Rate']))

	# Continue with model saving and performance plotting as before
	if phase % save_model_interval == 0:
		torch.save(agent.model.state_dict(), f"Models/{curr_dt}/model_phase_{phase + 1}.pth")
	if phase % save_video_interval == 0:
		save_videos(phase)
	print(
		f"End of Training Phase {phase + 1}\t|\tTotal Reward: {format(phase_total_score, '.2f')}\t|\tAverage Reward: {phase_total_score / (phase + 1)}")
	print("============== Starting Testing Phase ==============")
	for i in range(num_tests):
		testing_details += (test_agent(agent),)
	total_scores = [det[0] for det in testing_details]
	total_high_tile = [max([max(row) for row in det[1]]) for det in testing_details]
	total_moves_before_break = [len(det[1]) for det in testing_details]
	game_outputs = [det[2] for det in testing_details]
	break_count = sum([1 for det in game_outputs if det == 2])

	# Prepare data for tabulation
	for i, (score, high_tile, moves_before_break, game_output) in enumerate(
			zip(total_scores, total_high_tile, total_moves_before_break, game_outputs)):
		testing_output += (i + 1, score, high_tile, moves_before_break,
		                   color("Win", fg="black", bg="green") if game_output == 1 else
		                   color("Loss", fg="black", bg="red") if game_output == 0 else
		                   color("Break", fg="black", bg="yellow"),)

	# Print test results using tabulate
	print(tabulate(testing_output, headers=['Game', 'Score', 'High Tile', 'Moves', 'Game Output']))

	average_score = numpy.average(total_scores)
	average_break_count = numpy.average(break_count) * 100
	average_high_tile = sum(total_high_tile) / num_tests
	average_moves_before_break = sum(total_moves_before_break) / num_tests
	average_points_per_move = average_score / average_moves_before_break
	agent.scheduler.step(-average_score)
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
	# After all episodes in a phase are complete, decide which to learn from
	# For simplicity, let's learn from top N% of episodes
	#episode_details.sort(key=lambda x: x[0], reverse=True)  # Sort episodes by score, descending
	top_episodes = training_details[:max(1, len(training_details) * percentage_kept // 100)]
	teach_agent(agent, top_episodes)

	print(tabulate([[phase + 1, average_score, average_high_tile, average_moves_before_break,
	                 sum(agent.losses[-1 * len(top_episodes):-1]) / len(top_episodes) if len(agent.losses) > 0 else 0,
	                 KEYS.repetition_allowance, performance_threshold]],
	               headers=['Phase', 'Average Score', 'High Tile', 'Moves Before Break', 'Loss', 'Repetition Allowance',
	                        'Performance Threshold']))

	if average_score > performance_threshold:
		KEYS.repetition_allowance = max(1, KEYS.repetition_allowance - 1)
		performance_threshold += (average_score - performance_threshold)  #* 1.25
		#agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
		# Print color green
		color(f"Performance threshold reached, adjusted performance threshold to {performance_threshold}", "black",
		      "green")
	else:
		performance_threshold *= 0.9999
		#agent.epsilon = min(agent.epsilon_max,
		#                    agent.epsilon * (1 / (agent.epsilon_decay + ((1 - agent.epsilon_decay) / 4))))  # Slightly increase epsilon if performance is below threshold
		# Print color red
		color("Performance threshold NOT reached, adjusted performance threshold to {performance_threshold}", "black",
		      "red")

	last_average_score = average_score
	print(
		f"End of Phase {phase + 1}\t|\tAdjusted Epsilon: {agent.epsilon}\t|\tLearning Rate: {agent.optimizer.param_groups[0]['lr']}")
	print("=====================================================================")

print("==============\t|\tFinal Testing\t|\t==============")
torch.save(agent.model.state_dict(), f"Models/{curr_dt}/final_model.pth")
with multiprocessing.Pool(num_tests) as p:
	final_testing_details = p.map(test_agent, [agent for _ in range(
		num_tests)])  # final_testing_details = (score, game_output, (action, state, reward, done))

final_total_scores = [det[0] for det in final_testing_details]
final_high_tile = [max([max(row) for row in det[1]]) for det in final_testing_details]
final_moves_before_break = [len(det[1]) for det in final_testing_details]
final_game_outputs = [det[2] for det in final_testing_details]
final_break_count = sum([1 for det in final_game_outputs if det == 2])
print(tabulate(final_testing_details, headers=['Game', 'Score', 'High Tile', 'Moves', 'Game Output']))
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

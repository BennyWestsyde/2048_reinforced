import math

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Game import Game
import random

game = Game()


gamma = 1  # discounting rate
rewardSize = -1
gridSize = 4
terminationStates = [game.game_over(), game.checkWin()]
actions = [game.left, game.right, game.up, game.down]
numIterations = 10000
epsilon = 0.9  # exploration constant
min_epsilon = 0.1
max_epsilon = 1.0
decay_rate = 0.0001

def perform_random_action():
  index = np.random.randint(len(actions))
  return actions[index]

def perform_epsilon_greedy_action(game_instance, state, epsilon):
  random_value = np.random.random()
  if random_value < epsilon:
    index = np.random.randint(len(actions))
    return actions[index]
  else:
    best_action = np.argmax([actionRewardProb(state, action) for action in
                            [game_instance.left, game_instance.right, game_instance.up, game_instance.down]])
    ac = [game_instance.left, game_instance.right, game_instance.up, game_instance.down]
    return ac[best_action]

actions_mapping = {game.left: "left", game.right: "right", game.up: "up", game.down: "down"}


def actionRewardProb(state, action):
  temp_board = Game()
  temp_board.board = state.copy()
  reward_if_action = 0

  # copy of the state before action
  before_action = [x[:] for x in state]

  action_name = actions_mapping[action]
  getattr(temp_board, action_name)()

  # copy of the state after action
  after_action = temp_board.board

  if temp_board.board == state:  # checking if the state has changed after the action
    reward_if_action -= 10
  else:
    reward_if_action = rewardSize

  if game.checkWin():
    reward_if_action += 50
  elif game.game_over():
    reward_if_action -= 50

  # reward tile merges
  for i in range(4):
    for j in range(4):
      if before_action[i][j] < after_action[i][j]:  # a merge occurred
        reward_if_action += math.log(after_action[i][j].value, 2) * 10

  # reward if max value is in a corner
  corner_max_vals = [after_action[i][j] for i in [0, 3] for j in [0, 3]]
  if max(corner_max_vals) == temp_board.getMaxValue():
    reward_if_action += 25

  return reward_if_action

def getNextStateReward(state, action):
    temp_board = Game()
    temp_board.board = state.copy()

    reward = actionRewardProb(state, action)

    action_name = actions_mapping[action]
    getattr(temp_board, action_name)()  # Call with getattr
    nextState = temp_board.board

    return nextState, reward

valueMap = {}
states = [game.board]
reward_history = []
deltas = []
policyMap = {}

# Your relevant code portion with modified "copyValueMap"

# assuming valueMap and copyValueMap are dictionaries with list keys
# Your relevant code portion with modified "copyValueMap"

# assuming valueMap and copyValueMap are dictionaries with list keys
# Initialization of variables, maps, etc. before the training loop

for it in tqdm(range(numIterations)):
    # This is the new simulation portion where actions are taken based on epsilon-greedy policy and states are collected
    # Reset the game to initial state at the start of each episode
    game.reset()  # Please ensure your Game class has a reset method to bring back the game to initial state
    states = []  # Clear the states list


    # Play the game until it's over
    while not game.game_over():  # Assuming game_over() is a method in the Game class that returns True when the game is over
        null_state = [[0 for i in range(4)] for j in range(4)]
        current_state = game.board.copy()
        if game.board == [[0 for i in range(4)] for j in range(4)]:  # If the game is over, break out of the loop
            break
        action = perform_epsilon_greedy_action(game, current_state, epsilon)
        action()  # Perform the action
        if game.board.copy() == null_state:
            print("Null state reached")
        else:
            states.append(current_state)
    new_states = []
    for state in states:
       if state != [[0 for i in range(4)] for j in range(4)]:
          new_states.append(state)
    states = new_states.copy()
    # Rest of the original loop follows...
    # Here you update valueMap with collected states after each game
    copyValueMap = {tuple(k): v for k, v in valueMap.items()}  # convert list keys to tuple
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * it)
    deltaState = []
    weightedRewards = 0
    for state in states:
        if state == str(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 2, 0), (0, 0, 0, 0))):
          print("Null state reached")
          continue
        weightedRewards = 0
        for action in actions:
            nextState, reward = getNextStateReward(state, action)
            nextState_tuple = tuple([tuple(i) for i in nextState])
            new_reward = actionRewardProb(state, action) * (
                reward + gamma * valueMap.get(nextState_tuple, 0))
            reward_history.append({"state": state, "action": action, "reward": new_reward})
            weightedRewards += new_reward
        # convert list to a tuple before using as dictionary key
        state_tuple = tuple([tuple(i) for i in state])   # convert nested list to tuple of tuples
        if state_tuple == (((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 2, 0), (0, 0, 0, 0)),):
          print("Null state reached")
          continue
        deltaState.append(np.abs(copyValueMap.get(state_tuple, 0) - weightedRewards))
        copyValueMap[state_tuple] = weightedRewards
    if deltaState:
      deltas.append(np.mean(deltaState))
    else:
      deltas.append(0)  # append 0 if deltaState is empty
    valueMap = copyValueMap
    if it % 100 == 0:
      top_states = sorted(valueMap, key=lambda x: valueMap.get(x) if valueMap.get(x) is not None else float('-inf'),
                          reverse=True)[:5]
      for state in top_states:
        policy = policyMap.get(state)
        print(f"State: {state}, Value: {valueMap.get(state) if valueMap.get(state) is not None else 'None'}, Best Action: {policy if policy else 'No Policy'}")

    if it in [0, 1, 2, 9, 99, numIterations - 1]:
      print("Iteration {}".format(it + 1))
      print(game)
      print("Epsilon: {}".format(epsilon))
      print("Rewards: {}".format(weightedRewards))
      print("")
      print("")
for state in states:
  if state == [[0 for i in range(4)] for j in range(4)]:  # If the game is over, break out of the loop
    continue
  qVals = []
  for action in actions:
    nextState, reward = getNextStateReward(state, action)  # pass state as an argument here
    nextState = tuple([tuple(i) for i in nextState])
    qVals.append(reward + gamma * copyValueMap.get(nextState, 0))
  policyMap[tuple(tuple(sub) for sub in state)] = actions[np.argmax(qVals)]

# Assuming policyMap is your final policy map
readable_policy = {}

for state, action in policyMap.items():
    action_name = action.__name__  # get the name of the method, not the method itself
    readable_policy[str(state)] = action_name  # convert state to string for readability

unique_policies = {}
for state, action in readable_policy.items():
    if action not in unique_policies.values():
        unique_policies[state] = action

print("Final Policy Map in readable format:")
for state, action in unique_policies.items():
    print("State:")
    stateTuple = eval(state)  # convert string back to tuple
    for row in stateTuple:
        for item in row:
            print(str(item).center(6), end="")
        print("")

    print("Action: {}\n".format(action))

plt.figure(figsize=(20, 10))
plt.plot(deltas)
plt.show()

'''
plt.figure(figsize=(40, 20))
plt.imshow(reward_history, cmap='hot', interpolation='nearest')
plt.show()
'''


"""
print("Final Value Map")
print(valueMap)
print("")
print("")
"""
# ...
'''
print("Final State-Action Values")
print("")
print("---------------------------")
for state in states:  # Iterate directly over states
  state_tuple = tuple(tuple(sub) for sub in state)  # converting list to tuple
  print("---------------------------")
  print("|", end="")
  if policyMap[state_tuple] == game.left:
    print("  <  |", end="")
  elif policyMap[state_tuple] == game.right:
    print("  >  |", end="")
  elif policyMap[state_tuple] == game.up:
    print("  ^  |", end="")
  else:
    print("  v  |", end="")
  print("")
print("---------------------------")
print("")
print("")
input("Press Enter to continue...")'''
'''
print("Final Policy")
print("")
for state in states:  # Iterate directly over states
  state_tuple = tuple(tuple(sub) for sub in state)  # converting list to tuple
  print("---------------------------")
  print("|", end="")
  if policyMap[state_tuple] == game.left:
    print("  <  |", end="")
  elif policyMap[state_tuple] == game.right:
    print("  >  |", end="")
  elif policyMap[state_tuple] == game.up:
    print("  ^  |", end="")
  else:
    print("  v  |", end="")
  print("")
print("---------------------------")
print("")
print("")
input("Press Enter to continue...")
'''
import pandas as pd
reward_df = pd.DataFrame(reward_history)
print(reward_df)
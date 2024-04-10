import math
import os
import sys
import time
from os import system

import torch

from Cell import Cell, EmptyCell
import random
from pynput import keyboard

debug = True


class Game:
	def __init__(self, seed=None, state=None):
		if state is not None:
			self.fromState(state)
		else:
			self.reset(seed)

	def reset(self, seed=None):
		self.board = [[EmptyCell() for _ in range(4)] for _ in range(4)]
		self.score = 0
		self.max_value = 0
		self.total_value = 0
		self.total_empty_cells = 14
		if seed is not None:
			self.seed = random.seed(seed)
		else:
			self.seed = random.seed()
		self.newCell()
		self.newCell()
		self.last_move = None
		return self.getState()

	def reward(self, prev_state, action, new_state):
		reward = 0
		if self.checkWin():
			reward += 50
		if self.game_over():
			reward += -10
		if prev_state['board'] == new_state['board']:
			reward += -0.25
		#if new_state['score'] > prev_state['score'] != 0:
		#	reward += math.log2(new_state['score']-prev_state['score'])
		if new_state['total_empty_cells'] < prev_state['total_empty_cells'] and new_state['total_empty_cells'] != 0:
			reward += prev_state['total_empty_cells'] - new_state['total_empty_cells']
		if new_state['max_value'] > prev_state['max_value']:
			reward += math.log2(new_state['max_value'])
		return reward

	def randomMove(self, count=1):
		moves = []
		for _ in range(count):
			random.seed()
			move = random.randint(0, 3)
			moves.append(move)
			self.step(move)
		return moves

	def step(self, action):
		action_mapping = {0: "left", 1: "right", 2: "up", 3: "down"}
		prev_state = self.getState()
		getattr(self, action_mapping[action])()
		self.updateKeys()
		reward = self.reward(prev_state, action, self.getState())
		return self.getState(), reward, self.game_over()

	def getRandomState(self):
		random.seed()
		self.reset()
		self.randomMove(random.randint(0, 100))
		return self.getState()

	def getState(self):
		flat_board = []
		for row_num, row in enumerate(self.board):
			for col_num, cell in enumerate(row):
				flat_board.append(cell.value)

		output = {
			"board": [[cell.value for cell in row] for row in self.board],
			"score": self.score,
			"max_value": self.max_value,
			"total_value": self.total_value,
			"total_empty_cells": self.total_empty_cells
			}
		return output

	def fromState(self, state):
		self.reset()
		self.board = [[Cell(cell) for cell in row] for row in state['board']]
		self.score = state['score']
		self.max_value = state['max_value']
		self.total_value = state['total_value']
		self.updateKeys()

	def getMaxValue(self):
		return self.max_value

	def getScore(self):
		return self.score

	def getEmptyTiles(self):
		return self.total_empty_cells

	def updateKeys(self):
		self.max_value = max([max([cell.value for cell in row]) for row in self.board])
		self.total_value = sum([sum([cell.value for cell in row]) for row in self.board])
		self.total_empty_cells = sum([sum([1 for cell in row if cell.value == 0]) for row in self.board])

	def game_over(self):
		if self.total_empty_cells == 0:
			rows = self.rows()
			for row_num, row in enumerate(rows):
				for cell_num, cell in enumerate(row):
					# Check right neighbor if not in the rightmost column
					if cell_num < 3 and cell == row[cell_num + 1]:
						return False

					# Check bottom neighbor if not in the bottommost row
					if row_num < 3 and cell == rows[row_num + 1][cell_num]:
						return False
		else:
			return False
		return True

	def checkWin(self):
		if self.max_value >= 2048:
			return True
		return False

	def rows(self):
		return [[Cell(self.board[i][j].value) for j in range(4)] for i in range(4)]

	def cols(self):
		return [[Cell(self.board[i][j].value) for i in range(4)] for j in range(4)]

	def fromRows(self, rows):
		if len(rows) != 4:
			raise ValueError('rows must be a list of 4 lists')
		if rows != self.rows():
			self.board = rows
			return True
		else:
			return False

	def fromCols(self, cols):
		if len(cols) != 4:
			raise ValueError('cols must be a list of 4 lists')
		if cols != self.cols():
			self.board = [[cols[i][j] for i in range(4)] for j in range(4)]
			return True
		else:
			return False

	def newCell(self):
		random.seed(self.seed)
		empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i][j].value == 0]
		if not empty_cells: raise Exception("No more empty cells!")
		row, col = random.choice(empty_cells)
		self.board[row][col] = Cell(2)
		self.updateKeys()

	def shift(self, array: list[type(Cell)], reverse=False):
		if len(array) != 4:
			raise ValueError('array must be a list of 4 cells')
		if all([cell.value == 0 for cell in array]):
			return array
		if reverse:
			array = array[::-1].copy()

		for i in range(3, 0, -1):
			if array[i] == 0:
				# If the current cell is empty, place it at the front of the list
				for j in range(i - 1, -1, -1):
					if array[j] != 0:
						array[i], array[j] = array[j], array[i]
						break

		for i in range(3, 0, -1):
			if array[i] != 0:
				for j in range(i - 1, -1, -1):
					if array[i] == array[j]:
						array[j], array[i] = self.merge(array[i], array[j])
						i -= 1
					break

		for i in range(3, 0, -1):
			if array[i] == 0:
				# If the current cell is empty, place it at the front of the list
				for j in range(i - 1, -1, -1):
					if array[j] != 0:
						array[i], array[j] = array[j], array[i]
						break

		if reverse:
			array = array[::-1].copy()
		self.updateKeys()
		return array

	def merge(self, source: Cell, destination: Cell):
		if source.value != destination.value:
			return source, destination
		# The cells are the same value and can be merged
		destination += source
		source = EmptyCell()
		self.score += destination.value
		return source, destination

	def left(self):
		rows = self.rows()
		for i, row in enumerate(rows):
			if all([cell.value == 0 for cell in row]):
				continue
			rows[i] = self.shift(row, reverse=True)
		if self.fromRows(rows):
			self.newCell()
		self.last_move = "left"

	def right(self):
		rows = self.rows()
		for i, row in enumerate(rows):
			if all([cell.value == 0 for cell in row]):
				continue
			rows[i] = self.shift(row, reverse=False)
		if self.fromRows(rows):
			self.newCell()
		self.last_move = "right"

	def up(self):
		cols = self.cols()
		for i, col in enumerate(cols):
			if all([cell.value == 0 for cell in col]):
				continue
			cols[i] = self.shift(col, reverse=True)
		if self.fromCols(cols):
			self.newCell()
		self.last_move = "up"

	def down(self):
		cols = self.cols()
		for i, col in enumerate(cols):
			if all([cell.value == 0 for cell in col]):
				continue
			cols[i] = self.shift(col, reverse=False)
		if self.fromCols(cols):
			self.newCell()
		self.last_move = "down"

	def render(self):
		if os.name == 'nt' and not debug:
			system("cls")
		elif os.name == 'posix' and not debug:
			system("clear")
		else:
			print("\n\n")
		print("Score: {}".format(self.getScore()))
		print("Max Value: {}".format(self.getMaxValue()))
		print("Empty Tiles: {}".format(self.getEmptyTiles()))
		print("Game Over: {}".format(self.game_over()))
		print("Win: {}".format(self.checkWin()))
		move_mapping = {"left": "←", "right": "→", "up": "↑", "down": "↓", None: "N/A"}
		print("Last Move: {}".format(move_mapping[self.last_move]))
		print("Board:")
		row_num = 0
		for row in self.board:
			self.render_row(row_num, row)
			row_num += 1
		print()

	def render_row(self, row_num, row):
		output_row1 = ""
		output_row2 = ""
		output_row3 = ""
		output_row4 = ""
		for cell_num, cell in enumerate(row):
			if cell_num == 0 and row_num == 0:  # Top left corner
				output_row1 += "╔══════╦"
				output_row2 += f"║{cell.color}      " + "\33[0m║"
				output_row4 += f"║{cell.color}      " + "\33[0m║"
			elif cell_num == 0 and row_num == 3:  # Bottom left corner
				output_row1 += "╠══════╬"
				output_row2 += f"║{cell.color}      " + "\33[0m║"
				output_row4 += f"║{cell.color}      " + "\33[0m║"
			elif cell_num == 0:  # Left side
				output_row1 += "╠══════╬"
				output_row2 += f"║{cell.color}      " + "\33[0m║"
				output_row4 += f"║{cell.color}      " + "\33[0m║"
			elif cell_num == 3 and row_num == 0:  # Top right corner
				output_row1 += "══════╗"
				output_row2 += f"{cell.color}      \33[0m║"
				output_row4 += f"{cell.color}      \33[0m║"
			elif cell_num == 3 and row_num == 3:  # Bottom right corner
				output_row1 += "══════╣"
				output_row2 += f"{cell.color}      \33[0m║"
				output_row4 += f"{cell.color}      \33[0m║"
			elif cell_num == 3:  # Right side
				output_row1 += "══════╣"
				output_row2 += f"{cell.color}      \33[0m║"
				output_row4 += f"{cell.color}      \33[0m║"
			elif row_num == 0:  # Top side
				output_row1 += "══════╦"
				output_row2 += f"{cell.color}      \33[0m║"
				output_row4 += f"{cell.color}      \33[0m║"
			elif row_num == 3:  # Bottom side
				output_row1 += "══════╬"
				output_row2 += f"{cell.color}      \33[0m║"
				output_row4 += f"{cell.color}      \33[0m║"
			else:  # Middle
				output_row1 += "══════╬"
				output_row2 += f"{cell.color}      \33[0m║"
				output_row4 += f"{cell.color}      \33[0m║"
			if cell_num == 3:
				output_row3 += f"║{cell.color}{str(cell.value).center(6)}\33[0m║"
			else:
				output_row3 += f"║{cell.color}{str(cell.value).center(6)}\33[0m"
		print(output_row1)
		print(output_row2)
		print(output_row3)
		print(output_row4)
		if row_num == 3:
			print("╚══════╩══════╩══════╩══════╝")

	def __str__(self):
		return '\n'.join([' '.join([str(cell.value).center(6) for cell in row]) for row in self.board])

	def mainloop(self):
		self.render()
		while True:
			with keyboard.Events() as events:
				event = events.get(1e6)
				if event is None or event.key == keyboard.Key.esc or type(event) == keyboard.Events.Press:
					continue
				if event.key == keyboard.Key.left:
					self.left()
				elif event.key == keyboard.Key.right:
					self.right()
				elif event.key == keyboard.Key.up:
					self.up()
				elif event.key == keyboard.Key.down:
					self.down()
				else:
					continue
				print()
				self.render()
				if self.game_over():
					if self.checkWin():
						print("You Win!")
					else:
						print("Game Over!")
					break
				time.sleep(0.5)


if __name__ == '__main__':
	game = Game()
	game.mainloop()

import math

import Cell
import random
from pynput import keyboard


class Game:
  def __init__(self):
    self.reset()

  def reset(self):
    self.board = [[Cell.EmptyCell() for _ in range(4)] for _ in range(4)]
    self.keys = {
      "max_value": 0,
      "total_value": 0,
      "total_empty_cells": 0,
      "total_occupied_cells": 0,
      }
    self.newCell()
    self.newCell()

  def getHighestTile(self):
    return self.keys["max_value"]

  def getScore(self):
    return self.keys["total_value"]

  def getEmptyTiles(self):
    return self.keys["total_empty_cells"]

  def updateKeys(self):
    self.keys["max_value"] = max([max([cell.value for cell in row]) for row in self.board])
    self.keys["total_value"] = sum([sum([cell.value for cell in row]) for row in self.board])
    self.keys["total_empty_cells"] = sum([sum([1 for cell in row if cell.value == 0]) for row in self.board])
    self.keys["total_occupied_cells"] = 16 - self.keys["total_empty_cells"]

  def perform_action(self, action):
    """
    Perform the given action on the game board and return the new game state
    """
    # define your actions in a dictionary for easy calling
    actions_dict = {'left': self.left, 'right': self.right, 'up': self.up, 'down': self.down}

    # check if the action is valid
    if action in actions_dict:
      new_board_state = actions_dict[action]()

      # update the board with the new state
      self.board = new_board_state
      self.updateKeys()
    return self

  def game_over(self):
    if self.keys["total_empty_cells"] == 0:
      for row in self.board:
        for i in range(3):
          if row[i].value == row[i + 1].value:
            return False
    elif self.keys["max_value"] < 2048:
      return False
    return True

  def checkWin(self):
    if self.keys["max_value"] == 2048:
      return True
    return False

  def getPoints(self):
    totalvalue = self.keys["total_value"]
    totaloccupiedcells = self.keys["total_occupied_cells"]
    valuepoints = math.log(totalvalue, 2)
    cellpoints = math.log(totaloccupiedcells, 2)
    return valuepoints + cellpoints

  def rows(self):
    return [[self.board[i][j] for j in range(4)] for i in range(4)]

  def cols(self):
    return [[self.board[i][j] for i in range(4)] for j in range(4)]

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
    empty_cells = [(i, j) for i in range(4) for j in range(4)
                   if self.board[i][j].value == 0]
    if not empty_cells: raise Exception("No more empty cells!")
    row, col = random.choice(empty_cells)
    self.board[row][col] = Cell.Cell(2)
    self.updateKeys()

  def left(self):
    rows = self.rows()
    for row in rows:
      for i in range(3):
        if row[i].value == 0:
          for j in range(i + 1, 4):
            if row[j].value != 0:
              row[i], row[j] = row[j], row[i]
              break
    for row in rows:
      for i in range(3):
        row[i], row[i + 1] = row[i].merge(row[i + 1])
    if self.fromRows(rows):
      self.newCell()
    return self.board

  def right(self):
    rows = self.rows()
    for row in rows:
      for i in range(3, 0, -1):
        if row[i].value == 0:
          for j in range(i - 1, -1, -1):
            if row[j].value != 0:
              row[i], row[j] = row[j], row[i]
              break
    for row in rows:
      for i in range(3, 0, -1):
        row[i], row[i - 1] = row[i].merge(row[i - 1])
    if self.fromRows(rows):
      self.newCell()
    return self.board

  def up(self):
    cols = self.cols()
    for col in cols:
      for i in range(4):
        if col[i].value == 0:
          for j in range(i + 1, 4):
            if col[j].value != 0:
              col[i], col[j] = col[j], col[i]
              break
    for col in cols:
      for i in range(3):
        col[i], col[i + 1] = col[i].merge(col[i + 1])
    if self.fromCols(cols):
      self.newCell()
    return self.board

  def down(self):
    cols = self.cols()
    for col in cols:
      for i in range(3, -1, -1):
        if col[i].value == 0:
          pass
        else:
          for j in range(i + 1, 4):
            if col[j].value == 0:
              col[i], col[j] = col[j], col[i]
              break
    for col in cols:
      for i in range(3, 0, -1):
        col[i], col[i - 1] = col[i].merge(col[i - 1])
    if self.fromCols(cols):
      self.newCell()
    return self.board

  def __str__(self):
    return '\n'.join([' '.join([str(cell.value).center(6) for cell in row]) for row in self.board])


  def mainloop(self):
    print(self)
    while True:
      with keyboard.Events() as events:
        event = events.get(1e6)
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
        print(self)
        if self.game_over():
          print("Game Over!")
          break


if __name__ == '__main__':
  game = Game()
  game.mainloop()

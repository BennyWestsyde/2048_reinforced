import math
import time
from os import system

from Cell import Cell, EmptyCell
import random
from pynput import keyboard


class Game:
    def __init__(self, seed=None):
        self.reset(seed)

    def reset(self, seed=None):
        self.board = [[EmptyCell() for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.max_value = 0
        self.total_value = 0
        self.total_empty_cells = 14
        self.newCell()
        self.newCell()

    def step(self, action):
        action_mapping = {0: "left", 1: "right", 2: "up", 3: "down"}
        getattr(self, action_mapping[action])()
        return self.getState()

    def getState(self):
        flat_board = [cell.value for row in self.board for cell in row]
        output = {
            "board": flat_board,
            "score": self.score,
            "max_value": self.max_value,
            "total_value": self.total_value,
            "total_empty_cells": self.total_empty_cells,
            "game_over": self.game_over(),
            "win": self.checkWin()
            }
        return output
    

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
            for row in self.board:
                for i in range(3):
                    if row[i].value == row[i + 1].value:
                        return False
        elif self.max_value < 2048:
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

    def newCell(self, seed=None):
        if seed is not None:
            random.seed(seed)
        empty_cells = [(i, j) for i in range(4) for j in range(4)
                                     if self.board[i][j].value == 0]
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
        for i in range(3):
            if array[i] != 0:
                for j in range(i + 1, 4):
                    if array[j] == 0:
                        array[i], array[j] = array[j], array[i]
                        j += 1
                        i += 1
        for i in range(3, 0, -1):
            if array[i] != 0:
                for j in range(i - 1, -1, -1):
                    if array[i] == array[j]:
                        array[j], array[i] = self.merge(array[i], array[j])
                    break
        if reverse:

            for i in range(3, 0, -1):
                if array[i] != 0:
                    continue
                else:
                    for j in range(i - 1, -1, -1):
                        if array[j] != 0:
                            array[i], array[j] = array[j], array[i]
                            break
        else:
            for i in range(3):
                if array[i] == 0:
                    continue
                else:
                    for j in range(i + 1, 4):
                        if array[j] == 0:
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

    def right(self):
        rows = self.rows()
        for i, row in enumerate(rows):
            if all([cell.value == 0 for cell in row]):
                continue
            rows[i] = self.shift(row, reverse=False)
        if self.fromRows(rows):
            self.newCell()

    def up(self):
        cols = self.cols()
        for i, col in enumerate(cols):
            if all([cell.value == 0 for cell in col]):
                continue
            cols[i] = self.shift(col, reverse=True)
        if self.fromCols(cols):
            self.newCell()

    def down(self):
        cols = self.cols()
        for i, col in enumerate(cols):
            if all([cell.value == 0 for cell in col]):
                continue
            cols[i] = self.shift(col, reverse=False)
        if self.fromCols(cols):
            self.newCell()
        
    def render(self):
        if system == "Windows":
            system("cls")
        else:
            system("clear")
        print("Score: {}".format(self.getScore()))
        print("Max Value: {}".format(self.getMaxValue()))
        print("Empty Tiles: {}".format(self.getEmptyTiles()))
        print("Game Over: {}".format(self.game_over()))
        print("Win: {}".format(self.checkWin()))
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
        output_row5 = ""
        for cell_num, cell in enumerate(row):
            if cell_num == 0 and row_num == 0: # Top left corner
                output_row1 += "╔══════╦"
                output_row2 += f"║{cell.color}      "+"\33[0m║"
                output_row4 += f"║{cell.color}      "+"\33[0m║"
                output_row5 += "╠══════╬"
            elif cell_num == 0 and row_num == 3: # Bottom left corner
                output_row1 += "╠══════╬"
                output_row2 += f"║{cell.color}      "+"\33[0m║"
                output_row4 += f"║{cell.color}      "+"\33[0m║"
                output_row5 += "╚══════╩"
            elif cell_num == 0: # Left side
                output_row1 += "╠══════╬"
                output_row2 += f"║{cell.color}      "+"\33[0m║"
                output_row4 += f"║{cell.color}      "+"\33[0m║"
                output_row5 += "╠══════╬"
            elif cell_num == 3 and row_num == 0: # Top right corner
                output_row1 += "══════╗"
                output_row2 += f"{cell.color}      \33[0m║"
                output_row4 += f"{cell.color}      \33[0m║"
                output_row5 += "══════╣"
            elif cell_num == 3 and row_num == 3: # Bottom right corner
                output_row1 += "══════╣"
                output_row2 += f"{cell.color}      \33[0m║"
                output_row4 += f"{cell.color}      \33[0m║"
                output_row5 += "══════╝"
            elif cell_num == 3: # Right side
                output_row1 += "══════╣"
                output_row2 += f"{cell.color}      \33[0m║"
                output_row4 += f"{cell.color}      \33[0m║"
                output_row5 += "══════╣"
            elif row_num == 0: # Top side
                output_row1 += "══════╦"
                output_row2 += f"{cell.color}      \33[0m║"
                output_row4 += f"{cell.color}      \33[0m║"
                output_row5 += "══════╬"
            elif row_num == 3: # Bottom side
                output_row1 += "══════╬"
                output_row2 += f"{cell.color}      \33[0m║"
                output_row4 += f"{cell.color}      \33[0m║"
                output_row5 += "══════╩"
            else: # Middle
                output_row1 += "══════╬"
                output_row2 += f"{cell.color}      \33[0m║"
                output_row4 += f"{cell.color}      \33[0m║"
                output_row5 += "══════╬"
            if cell_num == 3:
                output_row3 += f"║{cell.color}{str(cell.value).center(6)}\33[0m║"
            else:
                output_row3 += f"║{cell.color}{str(cell.value).center(6)}\33[0m"
        print(output_row1)
        print(output_row2)
        print(output_row3)
        print(output_row4)
        print(output_row5)

    def __str__(self):
        return '\n'.join([' '.join([str(cell.value).center(6) for cell in row]) for row in self.board])

    def mainloop(self):
        self.render()
        while True:
            with keyboard.Events() as events:
                event = events.get(1e6)
                if event is None:
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
                time.sleep(0.1)


if __name__ == '__main__':
    game = Game()
    game.mainloop()

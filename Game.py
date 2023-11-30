import math
import time

from Cell import Cell, EmptyCell
import random
from pynput import keyboard


class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[EmptyCell() for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.max_value = 0
        self.total_value = 0
        self.total_empty_cells = 14
        self.newCell()
        self.newCell()

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

    def newCell(self):
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
                        j+=1
                        i+=1
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
        return array

    def merge(self, source:Cell, destination:Cell):
        if source.value != destination.value:
            return source, destination
        # The cells are the same value and can be merged
        destination += source
        source = EmptyCell()
        self.score += destination.value
        self.total_empty_cells += 1
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


    def __str__(self):
        return '\n'.join([' '.join([str(cell.value).center(6) for cell in row]) for row in self.board])


    def mainloop(self):
        print(self)
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
                print(self)
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

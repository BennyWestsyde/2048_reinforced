import pytest
from Game import Game
from Cell import Cell

class TestGame:
    def test_reset(self):
        game = Game()
        game.reset()
        assert game.getEmptyTiles() == 14
        assert game.getMaxValue() == 2
        assert game.getScore() == 0
        assert game.game_over() == False
        assert game.checkWin() == False

    def test_game_over(self):
        game = Game()
        game.reset()
        assert game.game_over() == False
        game.board = [[Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)],
                      [Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)]]
        game.updateKeys()
        assert game.game_over() == True

    def test_checkWin(self):
        game = Game()
        game.reset()
        assert game.checkWin() == False
        game.board = [[Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)],
                      [Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2048)]]
        game.updateKeys()
        assert game.checkWin() == True

    def test_rows(self):
        game = Game()
        game.reset()
        game.board = [[Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)],
                      [Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)]]
        assert game.rows() == [[Cell(2), Cell(4), Cell(8), Cell(16)],
                               [Cell(16), Cell(8), Cell(4), Cell(2)],
                               [Cell(2), Cell(4), Cell(8), Cell(16)],
                               [Cell(16), Cell(8), Cell(4), Cell(2)]]
        
    def test_cols(self):
        game = Game()
        game.reset()
        game.board = [[Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)],
                      [Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)]]
        assert game.cols() == [[Cell(2), Cell(16), Cell(2), Cell(16)],
                               [Cell(4), Cell(8), Cell(4), Cell(8)],
                               [Cell(8), Cell(4), Cell(8), Cell(4)],
                               [Cell(16), Cell(2), Cell(16), Cell(2)]]  
        
    def test_fromRows(self):
        game = Game()
        game.reset()
        game.board = [[Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)],
                      [Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)]]
        assert game.fromRows([[Cell(2), Cell(4), Cell(8), Cell(16)],
                               [Cell(16), Cell(8), Cell(4), Cell(2)],
                               [Cell(2), Cell(4), Cell(8), Cell(16)],
                               [Cell(16), Cell(8), Cell(4), Cell(2)]]) == False
        assert game.fromRows([[Cell(2), Cell(4), Cell(8), Cell(16)],
                               [Cell(16), Cell(0), Cell(0), Cell(2)],
                               [Cell(2), Cell(0), Cell(0), Cell(16)],
                               [Cell(16), Cell(8), Cell(4), Cell(4)]]) == True
        assert game.board == [[Cell(2), Cell(4), Cell(8), Cell(16)],
                               [Cell(16), Cell(0), Cell(0), Cell(2)],
                               [Cell(2), Cell(0), Cell(0), Cell(16)],
                               [Cell(16), Cell(8), Cell(4), Cell(4)]]
        
    def test_getMaxValue(self):
        game = Game()
        game.reset()
        assert game.getMaxValue() == 2
        game.board = [[Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2)],
                      [Cell(2), Cell(4), Cell(8), Cell(16)],
                      [Cell(16), Cell(8), Cell(4), Cell(2048)]]
        game.updateKeys()
        assert game.getMaxValue() == 2048
    
    def test_LeftShift(self):
        game = Game()
        game.reset()
        #assert game.shift([Cell(2), Cell(0), Cell(0), Cell(2)], reverse=False) == [Cell(0), Cell(0), Cell(0), Cell(4)]
        assert game.shift([Cell(2), Cell(2), Cell(2), Cell(2)], reverse=False) == [Cell(0), Cell(0), Cell(4), Cell(4)]
        assert game.shift([Cell(2), Cell(2), Cell(0), Cell(2)], reverse=False) == [Cell(0), Cell(0), Cell(2), Cell(4)]
        assert game.shift([Cell(2), Cell(2), Cell(4), Cell(2)], reverse=False) == [Cell(0), Cell(4), Cell(4), Cell(2)]
        assert game.shift([Cell(0), Cell(2), Cell(0), Cell(0)], reverse=False) == [Cell(0), Cell(0), Cell(0), Cell(2)]

    def test_RightShift(self):
        game = Game()
        game.reset()
        assert game.shift([Cell(2), Cell(0), Cell(0), Cell(2)], reverse=True) == [Cell(4), Cell(0), Cell(0), Cell(0)]
        assert game.shift([Cell(2), Cell(2), Cell(2), Cell(2)], reverse=True) == [Cell(4), Cell(4), Cell(0), Cell(0)]
        assert game.shift([Cell(2), Cell(2), Cell(0), Cell(2)], reverse=True) == [Cell(4), Cell(2), Cell(0), Cell(0)]
        assert game.shift([Cell(2), Cell(2), Cell(4), Cell(2)], reverse=True) == [Cell(4), Cell(4), Cell(2), Cell(0)]
        assert game.shift([Cell(0), Cell(2), Cell(0), Cell(0)], reverse=True) == [Cell(2), Cell(0), Cell(0), Cell(0)]

    def test_merge(self):
        game = Game()
        game.reset()
        assert game.merge(Cell(0), Cell(0)) == (Cell(0), Cell(0))
        assert game.merge(Cell(2), Cell(0)) == (Cell(2), Cell(0))
        assert game.merge(Cell(0), Cell(2)) == (Cell(0), Cell(2))
        assert game.merge(Cell(2), Cell(2)) == (Cell(0), Cell(4))
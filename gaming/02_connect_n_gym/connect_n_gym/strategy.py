import copy
import math
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple

from connect_n_gym.connect_n import ConnectNGame


class Strategy(ABC):

    def __init__(self, game):
        self.game = copy.deepcopy(game)
        super().__init__()

    @abstractmethod
    def action(self):
        pass


class MinimaxStrategy(Strategy):
    def action(self):
        result, move = self.minimax()
        # print(f'{result} {move}')
        return move

    def minimax(self) -> Tuple[int, Tuple[int, int]]:
        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.action(*pos)
                if result is None:
                    assert not game.gameOver
                    result, move = self.minimax()
                game.undo()
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == 1:
                    return 1, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.action(*pos)
                if result is None:
                    assert not game.gameOver
                    result, move = self.minimax()
                game.undo()
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == -1:
                    return -1, move
            return ret, bestMove


class MinimaxDPStrategy(Strategy):
    def action(self):
        result, move = self.minimax_dp(self.game.getStatus())
        print(f'{result} {move}')
        return move

    @lru_cache(maxsize=None)
    def minimax_dp(self, gameState) -> Tuple[int, Tuple[int, int]]:
        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.action(*pos)
                if result is None:
                    assert not game.gameOver
                    result, move = self.minimax_dp(game.getStatus())
                game.undo()
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == 1:
                    return 1, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.action(*pos)
                if result is None:
                    assert not game.gameOver
                    result, move = self.minimax_dp(game.getStatus())
                game.undo()
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if ret == -1:
                    return -1, move
            return ret, bestMove


class AlphaBetaStrategy(Strategy):

    def action(self):
        result, move = self.alpha_beta(self.game.getStatus(), -math.inf, math.inf)
        print(f'{result} {move}')
        return move

    def alpha_beta(self, status, alpha=None, beta=None) -> Tuple[int, Tuple[int, int]]:
        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.action(*pos)
                if result is None:
                    assert not game.gameOver
                    result, move = self.alpha_beta(game.getStatus(), alpha, beta)
                game.undo()
                alpha = max(alpha, result)
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == 1:
                    return ret, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.action(*pos)
                if result is None:
                    assert not game.gameOver
                    result, move = self.alpha_beta(game.getStatus(), alpha, beta)
                game.undo()
                beta = min(beta, result)
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == -1:
                    return ret, move
            return ret, bestMove


class AlphaBetaDPStrategy(Strategy):

    def action(self):
        self.alphaBetaStack = [(-math.inf, math.inf)]
        result, move = self.alpha_beta_dp(self.game.getStatus())
        print(f'{result} {move}')
        return move

    @lru_cache(maxsize=None)
    def alpha_beta_dp(self, status) -> Tuple[int, Tuple[int, int]]:
        alpha, beta = self.alphaBetaStack[-1]
        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.action(*pos)
                if result is None:
                    assert not game.gameOver
                    self.alphaBetaStack.append((alpha, beta))
                    result, move = self.alpha_beta_dp(game.getStatus())
                    self.alphaBetaStack.pop()
                game.undo()
                alpha = max(alpha, result)
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == 1:
                    return ret, move
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.action(*pos)
                if result is None:
                    assert not game.gameOver
                    self.alphaBetaStack.append((alpha, beta))
                    result, move = self.alpha_beta_dp(game.getStatus())
                    self.alphaBetaStack.pop()
                game.undo()
                beta = min(beta, result)
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
                if alpha >= beta or ret == -1:
                    return ret, move
            return ret, bestMove

if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(N=3, board_size=3)
    strategy = MinimaxStrategy(tic_tac_toe)
    strategy1 = MinimaxDPStrategy(tic_tac_toe)
    strategy2 = AlphaBetaStrategy(tic_tac_toe)
    strategy3 = AlphaBetaDPStrategy(tic_tac_toe)
    # print(strategy.action())
    # print('d')
    print(strategy3.action())

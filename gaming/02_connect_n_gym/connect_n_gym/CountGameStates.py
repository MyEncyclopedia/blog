import math
from typing import Tuple

from connect_n_gym.connect_n import ConnectNGame
from connect_n_gym.strategy import Strategy


class CountMinimaxDPStrategy(Strategy):
    def action(self, game) -> Tuple[int, Tuple[int, int]]:
        self.game = game
        self.dp = {}
        result, move = self.minimax_dp(self.game.getStatus())
        return result, move

    def minimax_dp(self, gameState) -> Tuple[int, Tuple[int, int]]:
        if gameState in self.dp:
            return self.dp[gameState]

        game = self.game
        bestMove = None
        assert not game.gameOver
        if game.currentPlayer == ConnectNGame.PLAYER_A:
            ret = -math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.minimax_dp(game.getStatus())
                else:
                    self.dp[game.getStatus()] = ret, bestMove
                game.undo()
                ret = max(ret, result)
                bestMove = move if ret == result else bestMove
            self.dp[gameState] = ret, bestMove
            return ret, bestMove
        else:
            ret = math.inf
            for pos in game.getAvailablePositions():
                move = pos
                result = game.move(*pos)
                if result is None:
                    assert not game.gameOver
                    result, oppMove = self.minimax_dp(game.getStatus())
                else:
                    self.dp[game.getStatus()] = ret, bestMove
                game.undo()
                ret = min(ret, result)
                bestMove = move if ret == result else bestMove
            self.dp[gameState] = ret, bestMove
            return ret, bestMove



if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(N=3, board_size=3)
    strategy = CountMinimaxDPStrategy()
    print(strategy.action(tic_tac_toe))
    print(len(strategy.dp))

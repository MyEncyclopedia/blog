import math
from functools import lru_cache

from connect_n_gym.connect_n import ConnectNGame


def minimax(game: ConnectNGame, isMaxPlayer: bool) -> int:
    """

    :param game:
    :param isMaxPlayer:
    :return: 1, 0, -1
    """
    assert not game.gameOver
    if isMaxPlayer:
        ret = -math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameOver
                result = minimax(game, not isMaxPlayer)
            game.undo()
            ret = max(ret, result)
            if ret == 1:
                return 1
        return ret
    else:
        ret = math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameOver
                result = minimax(game, not isMaxPlayer)
            game.undo()
            ret = min(ret, result)
            if ret == -1:
                return -1
        return ret

@lru_cache(maxsize=None)
def minimax_dp(game: ConnectNGame, gameState) -> int:
    """

    :param game:
    :param isMaxPlayer:
    :return: 1, 0, -1
    """
    assert not game.gameOver
    if game.currentPlayer == ConnectNGame.PLAYER_A:
        ret = -math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameOver
                result = minimax_dp(game, game.getStatus())
            game.undo()
            ret = max(ret, result)
            if ret == 1:
                return 1
        return ret
    else:
        ret = math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameOver
                result = minimax_dp(game, game.getStatus())
            game.undo()
            ret = min(ret, result)
            if ret == -1:
                return -1
        return ret

# @lru_cache_selected(maxsize=None, excludes=['alpha', 'beta'])
def alpha_beta_dp(game: ConnectNGame, status, alpha = None, beta = None) -> int:
    assert not game.gameOver
    if game.currentPlayer == ConnectNGame.PLAYER_A:
        ret = -math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameOver
                result = alpha_beta_dp(game, game.getStatus(), alpha, beta)
            game.undo()
            alpha = max(alpha, result)
            ret = max(ret, result)
            if alpha >= beta or ret == 1:
                return ret
        return ret
    else:
        ret = math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameOver
                result = alpha_beta_dp(game, game.getStatus(), alpha, beta)
            game.undo()
            beta = min(beta, result)
            ret = min(ret, result)
            if alpha >= beta or ret == -1:
                return ret
        return ret

def alpha_beta(game: ConnectNGame, status, alpha = None, beta = None) -> int:
    assert not game.gameOver
    if game.currentPlayer == ConnectNGame.PLAYER_A:
        ret = -math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameOver
                result = alpha_beta(game, game.getStatus(), alpha, beta)
            game.undo()
            alpha = max(alpha, result)
            ret = max(ret, result)
            if alpha >= beta or ret == 1:
                return ret
        return ret
    else:
        ret = math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameOver
                result = alpha_beta(game, game.getStatus(), alpha, beta)
            game.undo()
            beta = min(beta, result)
            ret = min(ret, result)
            if alpha >= beta or ret == -1:
                return ret
        return ret
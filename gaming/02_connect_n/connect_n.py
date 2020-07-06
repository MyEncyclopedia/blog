import copy
import math
from functools import lru_cache
from typing import List, Tuple

# 2. alpha beta pruning
# dp
class ConnectNGame:

    PLAYER_A = 1
    PLAYER_B = -1
    AVAILABLE = 0
    RESULT_TIE = 0
    RESULT_A_WIN = 1
    RESULT_B_WIN = -1

    def __init__(self, N:int = 3, board_size:int = 3):
        assert N <= board_size
        self.N = N
        self.board_size = board_size
        self.board = [[ConnectNGame.AVAILABLE] * board_size for _ in range(board_size)]
        self.gameEnded = False
        self.gameResult = None
        self.currentPlayer = ConnectNGame.PLAYER_A
        self.remainingPosNum = board_size * board_size
        self.actionStack = []

    def action(self, r: int, c: int):
        """

        :param r:
        :param c:
        :return: None: game ongoing
        """
        assert self.board[r][c] == ConnectNGame.AVAILABLE
        self.board[r][c] = self.currentPlayer
        self.actionStack.append((r, c))
        self.remainingPosNum -= 1
        if self.checkWin(r, c):
            self.gameEnded = True
            self.gameResult = self.currentPlayer
            return self.currentPlayer
        if self.remainingPosNum == 0:
            self.gameEnded = True
            self.gameResult = ConnectNGame.RESULT_TIE
            return ConnectNGame.RESULT_TIE
        self.currentPlayer *= -1

    def undo(self):
        if len(self.actionStack) > 0:
            lastAction = self.actionStack.pop()
            r, c = lastAction
            self.board[r][c] = ConnectNGame.AVAILABLE
            self.currentPlayer = ConnectNGame.PLAYER_A if len(self.actionStack) % 2 == 0 else ConnectNGame.PLAYER_B
            self.remainingPosNum += 1
            self.gameEnded = False
            self.gameResult = None
        else:
            raise Exception('No lastAction')

    def checkWin(self, r: int, c: int) -> bool:
        north = self.getConnectedNum(r, c, -1, 0)
        south = self.getConnectedNum(r, c, 1, 0)

        east = self.getConnectedNum(r, c, 0, 1)
        west = self.getConnectedNum(r, c, 0, -1)

        south_east = self.getConnectedNum(r, c, 1, 1)
        north_west = self.getConnectedNum(r, c, -1, -1)

        north_east = self.getConnectedNum(r, c, -1, 1)
        south_west = self.getConnectedNum(r, c, 1, -1)

        if (north + south + 1 >= self.N) or (east + west + 1 >= self.N) or \
                (south_east + north_west + 1 >= self.N) or (north_east + south_west + 1 >= self.N):
            return True
        return False

    def getConnectedNum(self, r: int, c: int, dr: int, dc: int) -> int:
        player = self.board[r][c]
        result = 0
        i = 1
        while True:
            new_r = r + dr * i
            new_c = c + dc * i
            if 0 <= new_r < self.board_size and 0 <= new_c < self.board_size:
                if self.board[new_r][new_c] == player:
                    result += 1
                else:
                    break
            else:
                break
            i += 1
        return result

    def getAvailablePositions(self) -> List[Tuple[int, int]]:
        return [(i,j) for i in range(self.N) for j in range(self.N) if self.board[i][j] == ConnectNGame.AVAILABLE]

    def getStatus(self):
        return tuple([tuple(tic_tac_toe.board[i]) for i in range(3)])



# def minimax(game: ConnectNGame, isMaxPlayer: bool) -> int:
#     """
#
#     :param game:
#     :param isMaxPlayer:
#     :return: 1, 0, -1
#     """
#     assert not game.gameEnded
#     if isMaxPlayer:
#         ret = - math.inf
#         for pos in game.getAvailablePositions():
#             gameClone = copy.deepcopy(game)
#             result = gameClone.action(*pos)
#             if result is None:
#                 assert not gameClone.gameEnded
#                 result = minimax(gameClone, not isMaxPlayer)
#             ret = max(ret, result)
#             if ret == 1:
#                 return 1
#         return ret
#     else:
#         ret = math.inf
#         for pos in game.getAvailablePositions():
#             gameClone = copy.deepcopy(game)
#             result = gameClone.action(*pos)
#             if result is None:
#                 assert not gameClone.gameEnded
#                 result = minimax(gameClone, not isMaxPlayer)
#             ret = min(ret, result)
#             if ret == -1:
#                 return -1
#         return ret

def minimax(game: ConnectNGame, isMaxPlayer: bool) -> int:
    """

    :param game:
    :param isMaxPlayer:
    :return: 1, 0, -1
    """
    assert not game.gameEnded
    if isMaxPlayer:
        ret = -math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameEnded
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
                assert not game.gameEnded
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
    assert not game.gameEnded
    if game.currentPlayer == ConnectNGame.PLAYER_A:
        ret = -math.inf
        for pos in game.getAvailablePositions():
            result = game.action(*pos)
            if result is None:
                assert not game.gameEnded
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
                assert not game.gameEnded
                result = minimax_dp(game, game.getStatus())
            game.undo()
            ret = min(ret, result)
            if ret == -1:
                return -1
        return ret

if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(N=3, board_size=3)
    print(minimax(tic_tac_toe, True))
    print(minimax_dp(tic_tac_toe, tic_tac_toe.getStatus()))


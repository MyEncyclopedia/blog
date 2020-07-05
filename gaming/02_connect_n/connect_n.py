import copy
from typing import List, Tuple


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

    def action(self, r: int, c: int):
        """

        :param r:
        :param c:
        :return: None: game ongoing
        """
        assert self.board[r][c] == ConnectNGame.AVAILABLE
        self.board[r][c] = self.currentPlayer
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

    def checkWin(self, r: int, c: int) -> bool:
        north = self.getConnectedNum(r, c, -1, 0)
        south = self.getConnectedNum(r, c, 1, 0)

        east = self.getConnectedNum(r, c, 0, 1)
        west = self.getConnectedNum(r, c, 0, -1)

        south_east = self.getConnectedNum(r, c, 1, 1)
        north_west = self.getConnectedNum(r, c, -1, -1)

        north_east = self.getConnectedNum(r, c, -1, 1)
        south_west = self.getConnectedNum(r, c, 1, -1)

        if (north + south + 1 >= self.connect_num) or (east + west + 1 >= self.connect_num) or \
                (south_east + north_west + 1 >= self.connect_num) or (north_east + south_west + 1 >= self.connect_num):
            return True
        return False

    def getConnectedNum(self, r: int, c: int, dr: int, dc: int) -> int:
        player = self.board[r][c]
        result = 0
        i = 1
        while True:
            new_r = r + dr * i
            new_c = c + dc * i
            if 0 <= new_r < self.board_num and 0 <= new_c < self.board_num:
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


def recurse(game: ConnectNGame):
    assert not game.gameEnded
    finalResult = False
    for pos in game.getAvailablePositions():
        result = game.action(*pos)
        if result is None:
            gameClone = copy.deepcopy(game)
            result = not recurse(gameClone)
        finalResult = finalResult or result
    return finalResult



if __name__ == '__main__':
    tic_tac_toe = ConnectNGame(n=3, board_size=3)
    print(recurse(tic_tac_toe))


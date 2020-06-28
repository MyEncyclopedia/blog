
# AC
class Solution:
    from functools import lru_cache
    @lru_cache(maxsize=None)
    # currentTotal < desiredTotal
    def alpha_beta(self, status: int, currentTotal: int, isMaxPlayer: bool, alpha: int, beta: int) -> int:
        import math
        if status == self.allUsed:
            return 0  # draw: no winner

        if isMaxPlayer:
            value = -math.inf
            for i in range(1, self.maxChoosableInteger + 1):
                if not (status >> i & 1):
                    new_status = 1 << i | status
                    if currentTotal + i >= self.desiredTotal:
                        return 1  # shortcut
                    value = max(value, self.alpha_beta(new_status, currentTotal + i, not isMaxPlayer, alpha, beta))
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        return value
            return value
        else:
            value = math.inf
            for i in range(1, self.maxChoosableInteger + 1):
                if not (status >> i & 1):
                    new_status = 1 << i | status
                    if currentTotal + i >= self.desiredTotal:
                        return -1  # shortcut
                    value = min(value, self.alpha_beta(new_status, currentTotal + i, not isMaxPlayer, alpha, beta))
                    beta = min(beta, value)
                    if alpha >= beta:
                        return value
            return value


    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        self.maxChoosableInteger = maxChoosableInteger
        self.desiredTotal = desiredTotal
        self.allUsed = 0
        for i in range(1, maxChoosableInteger + 1):
            self.allUsed = 1 << i | self.allUsed

        return self.alpha_beta(0, 0, True, -1, 1) == 1



if __name__ == "__main__":
    s = Solution()
    print(s.canIWin(5, 50))
    # print(s.canIWin(4, 6))



class Solution:
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def recurse(self, status: int, currentTotal: int) -> bool:
        for i in range(1, self.maxChoosableInteger + 1):
            if not (status >> i & 1):
                new_status = 1 << i | status
                if currentTotal + i >= self.desiredTotal:
                    return True
                if not self.recurse(new_status, currentTotal + i):
                    return True
        return False


    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        self.maxChoosableInteger = maxChoosableInteger
        self.desiredTotal = desiredTotal

        sum = maxChoosableInteger * (maxChoosableInteger + 1) / 2
        if sum < desiredTotal:
            return False
        return self.recurse(0, 0)


if __name__ == "__main__":
    s = Solution()
    # print(s.canIWin(5, 50))
    print(s.canIWin(4, 6))

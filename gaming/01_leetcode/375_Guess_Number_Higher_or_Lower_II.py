

class Solution_BottomUp:
    def getMoneyAmount(self, n: int) -> int:
        import math
        dp = [[0] * (n+1) for _ in range(n+1)]
        for gap in range(1, n):
            for lo in range(1, n+1-gap):
                hi = lo + gap
                dp[lo][hi] = math.inf
                for x in range(lo, hi):
                    dp[lo][hi] = min(dp[lo][hi], x + max(dp[lo][x-1], dp[x+1][hi]))
        return dp[1][n]


class Solution:

    from functools import lru_cache
    @lru_cache(maxsize=None)
    def recurse(self, l: int, r: int):
        if l >= r:
            return 0
        if l + 1 == r:
            return l
        import math
        ret = math.inf
        for i in range(l, r+1):
            ret = min(ret, i + max(self.recurse(l, i-1), self.recurse(i+1, r)))
        return ret

    def getMoneyAmount(self, n: int) -> int:
        return self.recurse(1, n)

if __name__ == "__main__":
    s = Solution()
    s1 = Solution_BottomUp()
    print(s.getMoneyAmount(3))
    print(s1.getMoneyAmount(3))

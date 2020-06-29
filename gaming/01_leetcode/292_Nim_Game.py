
# TLE
# Time Complexity: O(x^n)
class Solution_BruteForce:

    def canWinNim(self, n: int) -> bool:
        if n <= 3:
            return True
        for i in range(1, 4):
            if not self.canWinNim(n - i):
                return True
        return False

# RecursionError: maximum recursion depth exceeded in comparison n=1348820612
# Time Complexity: O(N)
class Solution_DP:
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def canWinNim(self, n: int) -> bool:
        if n <= 3:
            return True
        for i in range(1, 4):
            if not self.canWinNim(n - i):
                return True
        return False


# AC
# Time Complexity: O(1)
class Solution:
    def canWinNim(self, n: int) -> bool:
        return not (n % 4 == 0)

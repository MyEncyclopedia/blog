from functools import lru_cache
from typing import List

class Solution:
    @lru_cache(maxsize=None)
    def minimax(self, l: int, r: int, isMaxPlayer: bool) -> int:
        if l == r:
            return self.nums[l]

        if isMaxPlayer:
            return max(self.nums[l] + self.minimax(l + 1, r, not isMaxPlayer), self.nums[r] + self.minimax(l, r - 1, not isMaxPlayer))
        else:
            return min(self.minimax(l + 1, r, not isMaxPlayer) - self.nums[l], self.minimax(l, r - 1, not isMaxPlayer) - self.nums[r])

    def PredictTheWinner(self, nums: List[int]) -> bool:
        self.nums = nums
        v = self.minimax(0, len(nums) - 1, True)
        return v >= 0



if __name__ == "__main__":
    nums = [1, 5, 2]
    s = Solution()
    print(s.PredictTheWinner(nums))


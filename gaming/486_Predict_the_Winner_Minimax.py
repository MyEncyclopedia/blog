
# AC
from functools import lru_cache
from typing import List

class Solution:
    # max_player: max(A - B)
    # min_player: min(A - B)
    @lru_cache(maxsize=None)
    def minimax(self, l: int, r: int, isMaxPlayer: bool) -> int:
        if l == r:
            return self.nums[l] * (1 if isMaxPlayer else -1)

        if isMaxPlayer:
            return max(
                self.nums[l] + self.minimax(l + 1, r, not isMaxPlayer),
                self.nums[r] + self.minimax(l, r - 1, not isMaxPlayer))
        else:
            return min(
                -self.nums[l] + self.minimax(l + 1, r, not isMaxPlayer),
                -self.nums[r] + self.minimax(l, r - 1, not isMaxPlayer))

    def PredictTheWinner(self, nums: List[int]) -> bool:
        self.nums = nums
        v = self.minimax(0, len(nums) - 1, True)
        return v >= 0



if __name__ == "__main__":
    nums = [1, 5, 2]
    nums = [949829,1462205,1862548,20538,8366111,5424892,7189089,9,5301221,5462245,0,2,4,9401420,4937008,1,9,7,4146539]

    s = Solution()
    print(s.PredictTheWinner(nums))


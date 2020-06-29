import math
from functools import lru_cache
from typing import List

class Solution:
    def alpha_beta(self, l: int, r: int, curr: int, isMaxPlayer: bool, alpha: int, beta: int) -> int:
        if l == r:
            return curr + self.nums[l] * (1 if isMaxPlayer else -1)

        if isMaxPlayer:
            ret = self.alpha_beta(l + 1, r, curr + self.nums[l], not isMaxPlayer, alpha, beta)
            alpha = max(alpha, ret)
            if alpha >= beta:
                return alpha
            ret = max(ret, self.alpha_beta(l, r - 1, curr + self.nums[r], not isMaxPlayer, alpha, beta))
            return ret
        else:
            ret = self.alpha_beta(l + 1, r, curr - self.nums[l], not isMaxPlayer, alpha, beta)
            beta = min(beta, ret)
            if alpha >= beta:
                return beta
            ret = min(ret, self.alpha_beta(l, r - 1, curr - self.nums[r], not isMaxPlayer, alpha, beta))
            return ret

    def PredictTheWinner(self, nums: List[int]) -> bool:
        self.nums = nums
        v = self.alpha_beta(0, len(nums) - 1, 0, True, -math.inf, math.inf)
        return v >= 0



if __name__ == "__main__":
    # nums = [1, 5, 2]
    nums = [949829,1462205,1862548,20538,8366111,5424892,7189089,9,5301221,5462245,0,2,4,9401420,4937008,1,9,7,4146539]
    s = Solution()
    print(s.PredictTheWinner(nums))


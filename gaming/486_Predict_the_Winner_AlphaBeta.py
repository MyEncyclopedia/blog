import math
from functools import lru_cache
from typing import List

class Solution:
    # max_player: max(A - B)
    # min_player: min(A - B)
    # @lru_cache(maxsize=None)
    def alpha_beta(self, l: int, r: int, isMaxPlayer: bool, alpha: int, beta: int) -> int:
        if l == r:
            return self.nums[l] * (1 if isMaxPlayer else -1)

        if isMaxPlayer:
            choiceLeft = self.nums[l] + self.alpha_beta(l + 1, r, not isMaxPlayer, alpha, beta)
            alpha = max(alpha, choiceLeft)
            if alpha >= beta:
                # print(f'{choiceLeft}: {l}-{r} {isMaxPlayer}')
                return alpha
            choiceRight = self.nums[r] + self.alpha_beta(l, r - 1, not isMaxPlayer, alpha, beta)
            # return max(choiceLeft, choiceRight)
            v = max(choiceLeft, choiceRight)
            # print(f'{v}: {l}-{r} {isMaxPlayer}')
            return v
        else:
            choiceLeft = -self.nums[l] + self.alpha_beta(l + 1, r, not isMaxPlayer, alpha, beta)
            beta = min(beta, choiceLeft)
            if alpha >= beta:
                # print(f'{choiceLeft}: {l}-{r} {isMaxPlayer}')
                return beta

            choiceRight = -self.nums[r] + self.alpha_beta(l, r - 1, not isMaxPlayer, alpha, beta)
            v = min(choiceLeft, choiceRight)
            # print(f'{v}: {l}-{r} {isMaxPlayer}')
            return v

    def alpha_beta_2(self, l: int, r: int, isMaxPlayer: bool, alpha: int, beta: int, pre: int) -> int:
        if l == r:
            return pre + self.nums[l] * (1 if isMaxPlayer else -1)

        if isMaxPlayer:
            ret = self.alpha_beta_2(l + 1, r, not isMaxPlayer, alpha, beta, pre + self.nums[l])
            alpha = max(alpha, ret)
            if alpha >= beta:
                return alpha
            ret = max(ret, self.alpha_beta_2(l, r - 1, not isMaxPlayer, alpha, beta, pre + self.nums[r]))
            return ret
        else:
            ret = self.alpha_beta_2(l + 1, r, not isMaxPlayer, alpha, beta, pre - self.nums[l])
            beta = min(beta, ret)
            if alpha >= beta:
                return beta
            ret = min(ret, self.alpha_beta_2(l, r - 1, not isMaxPlayer, alpha, beta, pre - self.nums[r]))
            return ret

    def PredictTheWinner(self, nums: List[int]) -> bool:
        self.nums = nums
        # v = self.alpha_beta(0, len(nums) - 1, True, -100000000, 100000000)
        v = self.alpha_beta_2(0, len(nums) - 1, True, -100000000, 100000000, 0)
        print(f'v={v}')
        return v >= 0



if __name__ == "__main__":
    # nums = [1, 5, 2]
    nums = [949829,1462205,1862548,20538,8366111,5424892,7189089,9,5301221,5462245,0,2,4,9401420,4937008,1,9,7,4146539]
    s = Solution()
    print(s.PredictTheWinner(nums))


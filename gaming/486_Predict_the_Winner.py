from typing import List

# TLE
# Time Complexity: O(x^n)
class Solution_BruteForce:
    def maxDiff(self, l: int, r:int) -> int:
        if l == r:
            return self.nums[l]
        return max(self.nums[l] - self.maxDiff(l + 1, r), self.nums[r] - self.maxDiff(l, r - 1))

    def PredictTheWinner(self, nums: List[int]) -> bool:
        self.nums = nums
        return self.maxDiff(0, len(nums) - 1) >= 0

# AC
# Time Complexity: O(n^2)
class Solution:
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def maxDiff(self, l: int, r:int) -> int:
        if l == r:
            return self.nums[l]
        return max(self.nums[l] - self.maxDiff(l + 1, r), self.nums[r] - self.maxDiff(l, r - 1))

    def PredictTheWinner(self, nums: List[int]) -> bool:
        self.nums = nums
        # return self.maxDiff(0, len(nums) - 1) >= 0
        return self.maxDiff(0, len(nums) - 1)



if __name__ == "__main__":
    nums = [1, 5, 2]
    nums = [949829,1462205,1862548,20538,8366111,5424892,7189089,9,5301221,5462245,0,2,4,9401420,4937008,1,9,7,4146539]

    s = Solution()
    print(s.PredictTheWinner(nums))


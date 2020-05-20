import numpy as np
from typing import List
import matplotlib.pyplot as plt


class Solution:
    def totalLessEq2(self, A: List[int], K: int) -> int:
        m = np.zeros((len(A), len(A)))
        for i in range(0, len(A)):
            for j in range(i, len(A) + 1):
                if self.distinct(A[i:j]) <= K:
                    print(f'{i} -> {j}')
                    m[i, i:j] = 100.0
                    # m[i,i] = 200

        plt.matshow(m)
        plt.show()


    def distinct(self, A: List[int]) -> int:
        s = set()
        for i in A:
            s.add(i)
        return len(s)

    def totalLessEq(self, A: List[int], K: int) -> int:
        c2n_map = {}
        ret = 0
        i_start = 0
        i_end = -1
        while True:
            # try i_end, until sliding window > K
            while len(c2n_map) <= K and i_end + 1 < len(A):
                char = A[i_end + 1]
                if char in c2n_map or len(c2n_map) < K:
                    i_end += 1
                    c2n_map.setdefault(char, 0)
                    c2n_map[char] += 1
                else:
                    break
            if i_end == len(A) - 1:
                # ret += i_end - i_start
                # i_start += 1
                break

            # inc i_start and count
            while len(c2n_map) == K:
                ret += i_end - i_start + 1
                char = A[i_start]
                c2n_map[char] -= 1
                if c2n_map[char] == 0:
                    c2n_map.pop(char)
                i_start += 1

        for i in range(i_start, i_end + 1):
            ret += i_end - i + 1

        return ret


if __name__ == "__main__":
    # print(Solution().totalLessEq([1,2,1,2,3], 1))
    print(Solution().totalLessEq2([1,2,1,2,3], 2)) # should be 12
    # print(Solution().subarraysWithKDistinct([1,2,1,3,4], 3)) # should be 3
    # print(Solution().subarraysWithKDistinct([1,2], 1)) # should be 3


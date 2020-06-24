import functools
import inspect
from functools import lru_cache
from typing import List

class _MEquals(object):
    def __init__(self, parent_l, parent_r, pid, id):
        self.parent_l = parent_l
        self.parent_r = parent_r
        self.pid = pid
        self.id = id

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0

def lru_cache_ignoring_argument(*args, **kwargs):
    lru_decorator = functools.lru_cache(*args, **kwargs)

    def decorator(f):
        @lru_decorator
        def helper(theself, l, r, box, **kwargs):
            return f(theself, l, r, box.parent_l, box.parent_r, box.pid, box.id, **kwargs)

        @functools.wraps(f)
        def function(theself, l, r, parent_l, parent_r, pid, id, **kwargs):
            box = _MEquals(parent_l, parent_r, pid, id)
            return helper(theself, l, r, box, **kwargs)

        return function

    return decorator


def my_lru_cache(*args, **kwargs):
    def decorator(f):
        @functools.lru_cache(*args)
        def function(*args, **kwargs):
            print(*args)
            return f(*args, **kwargs)

        return function

    return decorator

from graphviz import Digraph
g = Digraph('G', filename='dp.gv', format='svg')
nodes = []
def graph_lru_cache(*args, **kwargs):
    def decorator(f):
        @lru_cache_ignoring_argument(*args)
        # @functools.lru_cache(*args)
        def function(*args, **kwargs):
            _, l, r, parent_l, parent_r, pid, id = args
            node_label = f'[{l}, {r}]{id}'
            node_label_parent = f'[{parent_l}, {parent_r}]{pid}'
            if not node_label in nodes:
                nodes.append(node_label)
            print(f'{node_label} -> {node_label_parent}')
            g.edge(node_label, node_label_parent)

            return f(*args, **kwargs)

        return function

    return decorator

class Solution:

    # @my_lru_cache(maxsize=None)
    # def maxDiff(self, l: int, r:int) -> int:
    #     if l == r:
    #         return self.nums[l]
    #     return max(self.nums[l] - self.maxDiff(l + 1, r), self.nums[r] - self.maxDiff(l, r - 1))
    #
    # def PredictTheWinner(self, nums: List[int]) -> bool:
    #     self.nums = nums
    #     return self.maxDiff(0, len(nums) - 1) >= 0


    @graph_lru_cache(maxsize=None)
    def maxDiff(self, l: int, r:int, parent_l: int, parent_r: int, pid: int, id: int) -> int:
        self.id += 1
        if l == r:
            return self.nums[l]
        max1 = self.nums[l] - self.maxDiff(l + 1, r, l, r, id, self.id)
        self.id += 1
        max2 = self.nums[r] - self.maxDiff(l, r - 1, l, r, id, self.id)
        return max(max1, max2)

    def PredictTheWinner(self, nums: List[int]) -> bool:
        self.nums = nums
        self.id = 1
        return self.maxDiff(0, len(nums) - 1, 0, len(nums)-1, 0, self.id) >= 0


def run_graphviz():
    from graphviz import Digraph
    g = Digraph('G', filename='brute.gv', format='svg')
    g.edge('Hello', 'World')
    g.view()

def draw():
    nums = [1, 5, 2, 2]
    s = Solution()
    print(s.PredictTheWinner(nums))
    g.view()

if __name__ == "__main__":
    draw()
    # nums = [1, 5, 2]
    # s = Solution()
    # print(s.PredictTheWinner(nums))


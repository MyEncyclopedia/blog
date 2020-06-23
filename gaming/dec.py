import functools
import inspect
from functools import lru_cache
from typing import List


from itertools import chain
import traceback
import sys

import sys, traceback

def print_exc_plus(  ):
    """
    Print the usual traceback information, followed by a listing of all the
    local variables in each frame.
    """
    tb = sys.exc_info()[2]
    while 1:
        if not tb.tb_next:
            break
        tb = tb.tb_next
    stack = []
    f = tb.tb_frame
    while f:
        stack.append(f)
        f = f.f_back
    stack.reverse(  )
    traceback.print_exc(  )
    print("Locals by frame, innermost last")
    for frame in stack:
        print()
        print("Frame %s in %s at line %s" % (frame.f_code.co_name, frame.f_code.co_filename, frame.f_lineno))
        for key, value in frame.f_locals.items(  ):
            print("\t%20s = " % key)
            # We have to be VERY careful not to cause a new error in our error
            # printer! Calling str(  ) on an unknown object could cause an
            # error we don't want, so we must use try/except to catch it --
            # we can't stop it from happening, but we can and should
            # stop it from propagating if it does happen!
            try:
                print(value)
            except:
                print("<ERROR WHILE PRINTING VALUE>")

def stackdump(id='', msg='HERE'):
    # frames = inspect.trace()
    # argvalues = inspect.getargvalues(frames[0][0])
    # print(sss)
    # print('ENTERING STACK_DUMP' + (': '+id) if id else '')
    raw_tb = traceback.extract_stack()
    entries = traceback.format_list(raw_tb)

    frame = sys._getframe(1)
    sss = inspect.getargvalues(frame)

    # Remove the last two entries for the call to extract_stack() and to
    # the one before that, this function. Each entry consists of single
    # string with consisting of two lines, the script file path then the
    # line of source code making the call to this function.
    del entries[-2:]

    # Split the stack entries on line boundaries.
    lines = list(chain.from_iterable(line.splitlines() for line in entries))
    if msg:  # Append it to last line with name of caller function.
        lines[-1] += ' <-- ' + str(msg)
        # lines.append('LEAVING STACK_DUMP' + (': '+id) if id else '')
    print('\n'.join(lines))
    print()

sys.modules[__name__] = stackdump  # Make a callable module.

def lru_cache_ignoring_first_argument(*args, **kwargs):
    lru_decorator = functools.lru_cache(*args, **kwargs)

    def decorator(f):
        @lru_decorator
        def helper(arg1, *args, **kwargs):
            return f(arg1, *args, **kwargs)

        @functools.wraps(f)
        def function(arg1, *args, **kwargs):
            # print(arg1)
            print(*args)
            # print(**kwargs)
            stackdump(*args)
            # print_exc_plus()
            return helper(arg1, *args, **kwargs)

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

class Solution:

    # @mylru_cache(maxsize=None)
    # @multiple_decorators(maxsize=None)
    @lru_cache_ignoring_first_argument(maxsize=None)
    def maxDiff(self, l: int, r:int) -> int:
        if l == r:
            return self.nums[l]
        return max(self.nums[l] - self.maxDiff(l + 1, r), self.nums[r] - self.maxDiff(l, r - 1))

    def PredictTheWinner(self, nums: List[int]) -> bool:
        self.nums = nums
        return self.maxDiff(0, len(nums) - 1) >= 0




if __name__ == "__main__":
    nums = [1, 5, 2]
    s = Solution()
    print(s.PredictTheWinner(nums))
    # from graphviz import Digraph
    # g = Digraph('G', filename='hello.gv')
    # g.edge('Hello', 'World')
    # g.view()

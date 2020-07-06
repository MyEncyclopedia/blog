import inspect
import uuid


# todo
# multiple usage?
# kwargs mix args test
def lru_cache_selected(*decorator_args, **decorator_kwargs):
    # print(*decorator_args)
    excludes = decorator_kwargs.get('excludes', [])
    decorator_kwargs.pop('excludes')

    class _Equals(object):

        def __init__(self, *func_args, **func_kwargs):
            self.func_args = func_args
            self.func_kwargs = func_kwargs

        def __eq__(self, other):
            return True

        def __hash__(self):
            return 0

    import functools
    lru_decorator = functools.lru_cache(*decorator_args, **decorator_kwargs)

    def decorator(f):
        f_arg_spec = inspect.getfullargspec(f)
        # len_func_args = len(f_arg_spec.args) - len(f_arg_spec.defaults)
        # args_list = f_arg_spec.args

        def filter_cached_args(*func_args):
            return [func_args[i] for i in range(len(func_args)) if f_arg_spec.args[i] not in excludes]

        @lru_decorator
        def helper(box: _Equals, *effective_args, **effective_kwargs):
            return f(*box.func_args, **box.func_kwargs)

        @functools.wraps(f)
        def function(*func_args, **func_kwargs):
            exclude_kwargs = {}
            effective_args = filter_cached_args(*func_args)
            effective_kwargs = {k: v for k, v in func_kwargs.items() if k not in excludes}
            box = _Equals(*func_args, **func_kwargs)
            return helper(box, *effective_args, **effective_kwargs)

        return function

    return decorator

d = [[0] * 5 for i in range(6)]


# @lru_cache_selected(maxsize=None, excludes=['ex1', 'ex2'])
def d2(x, ex1, ex2=None, y=None):
    print(f'x={x}, y={y}, ex1={ex1}, ex2={ex2}')
    if x == 0 or y == 0:
        d[x][y] = x + y
        return d[x][y]
    if d[x][y] == 0:
        d2(x-1, str(uuid.uuid4()), str(uuid.uuid4()), y)
        d2(x, str(uuid.uuid4()), str(uuid.uuid4()), y-1)
        d[x][y] = d[x-1][y] * d[x][y-1]
        return d[x][y]
    else:
        return d[x][y]



if __name__ == "__main__":
    d2(5, str(uuid.uuid4()), str(uuid.uuid4()), 4)


# Note: this impl has constraint that all params in excludes must be named args (in func_kwargs)
# def lru_cache_selected_(*decorator_args, **decorator_kwargs):
#     # print(*decorator_args)
#     excludes = decorator_kwargs.get('excludes', [])
#     decorator_kwargs.pop('excludes')
#
#     class _Equals(object):
#
#         def __init__(self, **exc_kwargs):
#             self.exc = exc_kwargs
#
#         def __eq__(self, other):
#             return True
#
#         def __hash__(self):
#             return 0
#
#     import functools
#     lru_decorator = functools.lru_cache(*decorator_args, **decorator_kwargs)
#
#     def decorator(f):
#         b = inspect.getfullargspec(f)
#         print(b)
#         @lru_decorator
#         def helper(box: _Equals, *func_args, **cached_kwargs):
#             cached_kwargs.update(box.exc)
#             return f(*func_args, **cached_kwargs)
#
#         @functools.wraps(f)
#         def function(*func_args, **func_kwargs):
#             # all params in excludes must be named args (in func_kwargs)
#             exclude_kwargs = {k: v for k, v in func_kwargs.items() if k in excludes}
#             cached_kwargs = {k: v for k, v in func_kwargs.items() if k not in excludes}
#             box = _Equals(**exclude_kwargs)
#             return helper(box, *func_args, **cached_kwargs)
#
#         return function
#
#     return decorator

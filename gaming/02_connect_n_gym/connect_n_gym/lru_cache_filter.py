
# todo
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
        import inspect
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

@lru_cache_selected(excludes=[])
def demo_no_excludes(x: int, y:int) -> int:
    print(f'in demo_no_excludes {x}, {y}')
    return x + 1

@lru_cache_selected(excludes=['y'])
def demo_excludes_kwarg(x: int, y:int = None) -> int:
    print(f'in demo_excludes_kwarg {x}, {y}')
    return x + 1

@lru_cache_selected(excludes=['y', 'z'])
def demo_excludes_arg_and_kwarg(x: int, y, z:int = None) -> int:
    print(f'in demo_excludes_arg_and_kwarg {x}, {y}, {z}')
    return x + 1


if __name__ == "__main__":
    # print(demo_no_excludes(1, 2))
    # print(demo_no_excludes(3, 2))
    # print(demo_no_excludes(1, 2))
    #
    # print(demo_excludes_kwarg(1, 2))
    # print(demo_excludes_kwarg(3, 2))
    # print(demo_excludes_kwarg(1, 3))

    print(demo_excludes_arg_and_kwarg(1, 2, 3))
    print(demo_excludes_arg_and_kwarg(1, 3, 4))
    print(demo_excludes_arg_and_kwarg(2, 3, 4))


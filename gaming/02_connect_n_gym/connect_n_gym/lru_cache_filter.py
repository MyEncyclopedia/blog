
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
            args_len = len(func_args) - 0 if f_arg_spec.defaults is None else len(f_arg_spec.defaults)
            return [func_args[i] for i in range(args_len) if f_arg_spec.args[i] not in excludes]

        @lru_decorator
        def helper(box: _Equals, *effective_args, **effective_kwargs):
            return f(*box.func_args, **box.func_kwargs)

        @functools.wraps(f)
        def function(*func_args, **func_kwargs):
            print(f_arg_spec)
            exclude_kwargs = {}
            effective_args = filter_cached_args(*func_args)
            effective_kwargs = {k: v for k, v in func_kwargs.items() if k not in excludes}
            box = _Equals(*func_args, **func_kwargs)
            return helper(box, *effective_args, **effective_kwargs)

        return function

    return decorator

@lru_cache_selected(excludes=[])
def demo_no_excludes(x: str, y:str) -> str:
    print(f'in demo_no_excludes {x}, {y}')
    return x

@lru_cache_selected(excludes=['y'])
def demo_excludes_kwarg(x: str, y:str = None) -> str:
    print(f'in demo_excludes_kwarg {x}, {y}')
    return x

@lru_cache_selected(excludes=['y', 'z'])
def demo_excludes_arg_and_kwarg(x: str, y:str, z:str = None) -> str:
    print(f'in demo_excludes_arg_and_kwarg {x}, {y}, {z}')
    return x

@lru_cache_selected(excludes=['y', 'z'])
def demo_excludes_all_mixed(x: str, y:str, z:str = None, l:str = None) -> str:
    print(f'in demo_excludes_arg_and_kwarg {x}, {y}, {z}, {l}')
    return x

@lru_cache_selected(excludes=['y', 'z'])
def demo_excludes_disorder(x: str, z:str, y:str = None, l:str = None) -> str:
    print(f'in demo_excludes_disorder {x}, {y}, {z}, {l}')
    return x

if __name__ == "__main__":
    # print(demo_no_excludes(1, 2))
    # print(demo_no_excludes(3, 2))
    # print(demo_no_excludes(1, 2))
    #
    # print(demo_excludes_kwarg(1, 2))
    # print(demo_excludes_kwarg(3, 2))
    # print(demo_excludes_kwarg(1, 3))

    # print(demo_excludes_arg_and_kwarg(1, 2, 3))
    # print(demo_excludes_arg_and_kwarg(1, 3, 4))
    # print(demo_excludes_arg_and_kwarg(2, 3, 4))

    print(demo_excludes_all_mixed("x1", "y1", "z1", "l1"))
    print(demo_excludes_all_mixed("x1", "y1", "z1", "l1"))
    print(demo_excludes_all_mixed("x1", "y1", "z1", "l1"))

    print(demo_excludes_disorder("x1", "z1", l="l1", y="y1"))
    print(demo_excludes_disorder("x2", "z1", l="l1", y="y1"))
    print(demo_excludes_disorder("x2", "z1", l="l1", y="y2"))

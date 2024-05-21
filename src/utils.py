def function_builder(func, **kwargs):
    def inner(*inner_args, **inner_kwargs):
        return func(*inner_args, **kwargs, **inner_kwargs)

    inner.__name__ = func.__name__
    return inner

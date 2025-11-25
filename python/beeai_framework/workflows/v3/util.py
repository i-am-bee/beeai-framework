# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Awaitable, Callable
from typing import Any


async def run_callable(
    func: Callable[..., Any] | Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Dynamically inspects a callable (sync or async) and calls it with
    only the arguments it accepts.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Determine if the callable is bound
    is_bound = hasattr(func, "__self__") and func.__self__ is not None
    skip_first = False

    if params:
        first_param = params[0]
        if is_bound and first_param.name in ("self", "cls"):
            skip_first = True

    filtered_args = []
    filtered_kwargs = {}

    arg_index = 0
    for i, param in enumerate(params):
        if skip_first and i == 0:
            continue

        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if arg_index < len(args):
                filtered_args.append(args[arg_index])
                arg_index += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            filtered_args.extend(args[arg_index:])
            arg_index = len(args)
            break
        elif param.kind == inspect.Parameter.KEYWORD_ONLY or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if param.name in kwargs:
                filtered_kwargs[param.name] = kwargs[param.name]
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            filtered_kwargs = kwargs
            break

    # Bind safely
    try:
        bound = sig.bind_partial(*filtered_args, **filtered_kwargs)
        bound.apply_defaults()
    except TypeError:
        bound = sig.bind_partial(**filtered_kwargs)

    # Call
    if inspect.iscoroutinefunction(func):
        result = await func(*bound.args, **bound.kwargs)
    else:
        result = func(*bound.args, **bound.kwargs)

    return result

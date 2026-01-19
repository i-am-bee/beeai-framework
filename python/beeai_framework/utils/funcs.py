# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, Unpack, get_args, get_origin

import typing_extensions

from beeai_framework.utils.dicts import include_keys, is_typed_dict_type

T = TypeVar("T")
P = ParamSpec("P")


def identity(value: T) -> T:
    return value


def is_same_function(f1: Callable[..., Any], f2: Callable[..., Any]) -> bool:
    """Checks if two callables refer to the same function or method.

    This comparison handles regular functions and methods. For bound methods, it checks
    that both the underlying function and the instance they are bound to are identical.

    Args:
        f1: The first callable.
        f2: The second callable.

    Returns:
        True if the callables are considered the same, False otherwise.
    """
    # Simple functions
    if f1 is f2:
        return True

    func1 = getattr(f1, "__func__", f1)
    func2 = getattr(f2, "__func__", f2)

    # Compare the underlying function object
    if func1 is not func2:
        return False

    # If either is a bound method, also compare the bound "self"
    self1 = getattr(f1, "__self__", None)
    self2 = getattr(f2, "__self__", None)

    return self1 is self2


def safe_invoke(cls: Callable[P, T]) -> Callable[P, T]:
    allowed_kwargs = get_keyword_arg_names(cls)

    def construct(*args: P.args, **kwargs: P.kwargs) -> T:
        valid_kwargs = include_keys(kwargs, allowed_kwargs)
        # pyrefly: ignore [invalid-param-spec]
        return cls(*args, **valid_kwargs)

    return construct


def get_keyword_arg_names(func: Callable[..., Any]) -> set[str]:
    """
    Extract names of all keyword arguments of a function including **kwargs (if present).
    """
    signature = inspect.signature(func)
    keyword_args = []

    for name, param in signature.parameters.items():
        if param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD):
            keyword_args.append(name)
        elif param.kind == param.VAR_KEYWORD:
            annotation = param.annotation
            origin = get_origin(annotation)
            if origin is Unpack or origin is typing_extensions.Unpack:
                (inner,) = get_args(annotation)
                if is_typed_dict_type(inner):
                    # Extract the TypedDict keys
                    keyword_args.extend(inner.__annotations__.keys())
                else:
                    # Unpack of non-TypedDict, fallback to placeholder
                    keyword_args.append(f"**{name}")
            else:
                # Generic **kwargs
                keyword_args.append(f"**{name}")

    return set(keyword_args)

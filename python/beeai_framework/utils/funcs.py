# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def identity(value: T) -> T:
    return value


def is_same_function(f1: Callable[..., Any], f2: Callable[..., Any]) -> bool:
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

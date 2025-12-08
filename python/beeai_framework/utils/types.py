# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeAlias, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

MaybeAsync: TypeAlias = Callable[P, R] | Callable[P, Awaitable[R]]


primitive_types = (int, float, bool, str, bytes, complex, type(None))


def is_primitive(value: Any) -> bool:
    # if value is a type, check it directly (e.g. value == int)
    if isinstance(value, type):
        return value in primitive_types

    # if value is an instance, check it normally
    return isinstance(value, primitive_types)

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


def ensure_async(fn: Callable[P, T | Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    if asyncio.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        result: T | Awaitable[T] = await asyncio.to_thread(fn, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        else:
            return result

    return wrapper


async def to_async_generator(items: list[T]) -> AsyncGenerator[T]:
    for item in items:
        yield item

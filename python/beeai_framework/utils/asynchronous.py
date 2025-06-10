# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


def ensure_async(fn: Callable[..., T | Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    if asyncio.iscoroutinefunction(fn):
        return fn  # Already async, no wrapping needed.

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        result: T | Awaitable[T] = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    return wrapper


async def to_async_generator(items: list[T]) -> AsyncGenerator[T]:
    for item in items:
        yield item

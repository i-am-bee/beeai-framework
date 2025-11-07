# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import functools
import inspect
from asyncio import CancelledError
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

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


async def cancel_task(task: asyncio.Task[None] | None) -> None:
    if task:
        task.cancel()
        with contextlib.suppress(CancelledError):
            await task


def awaitable_to_coroutine(awaitable: Awaitable[T]) -> Coroutine[Any, Any, T]:
    async def as_coroutine() -> T:
        return await awaitable

    return as_coroutine()


def run_sync(awaitable: Awaitable[T], *, timeout: int | None = None) -> T:
    """
    Run *awaitable* from synchronous code.

    - If we're already inside the loop's thread, raise an error (to avoid dead-lock).
    - If no loop is running, create one temporarily.
    - If a loop is running in another thread, schedule thread-safely.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if asyncio.iscoroutine(awaitable):
            return asyncio.run(awaitable, debug=False)  # type: ignore[no-any-return]
        else:
            return asyncio.run(awaitable_to_coroutine(awaitable), debug=False)

    if loop.is_running() and loop == asyncio.get_running_loop():
        raise RuntimeError("blocking_await() called from inside the event-loop thread; would dead-lock")

    if asyncio.iscoroutine(awaitable):
        fut = asyncio.run_coroutine_threadsafe(awaitable, loop)
    else:
        fut = asyncio.run_coroutine_threadsafe(awaitable_to_coroutine(awaitable), loop)
    return fut.result(timeout=timeout)  # type: ignore[no-any-return]

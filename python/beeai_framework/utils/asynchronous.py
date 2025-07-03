# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import functools
import inspect
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


def awaitable_to_coroutine(awaitable: Awaitable[T]) -> Coroutine[Any, Any, T]:
    async def as_coroutine() -> T:
        return await awaitable

    return as_coroutine()


def sync_run_awaitable(awaitable: Awaitable[T], timeout: int | None = None) -> T:
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
            return asyncio.run(awaitable, debug=False)
        else:
            return asyncio.run(awaitable_to_coroutine(awaitable), debug=False)

    if loop.is_running() and loop == asyncio.get_running_loop():
        raise RuntimeError("blocking_await() called from inside the event-loop thread; would dead-lock")

    fut = asyncio.run_coroutine_threadsafe(awaitable, loop)
    return fut.result(timeout=timeout)

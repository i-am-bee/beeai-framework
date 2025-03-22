# Copyright 2025 IBM Corp.
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

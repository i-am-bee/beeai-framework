import asyncio
import sys
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

from beeai_framework.cache import BaseCache, SlidingCache
from beeai_framework.errors import FrameworkError

P = ParamSpec("P")
R = TypeVar("R")


def cached(
    cache: SlidingCache[R],
    *,
    enabled: bool = True,
    key_fn: Callable[[tuple[Any, ...], dict[str, Any]], str] | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Basic async caching decorator that reuses SlidingCache."""

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not enabled:
                return await fn(*args, **kwargs)

            key = key_fn(args, kwargs) if key_fn else BaseCache.generate_key({"args": args, "kwargs": kwargs})
            cached_value = await cache.get(key)
            if cached_value is not None or await cache.has(key):
                return cached_value  # type: ignore[return-value]

            result = await fn(*args, **kwargs)
            await cache.set(key, result)
            return result

        return wrapper

    return decorator


request_cache: SlidingCache[str] = SlidingCache(size=8, ttl=2)


class ReportGenerator:
    def __init__(self) -> None:
        self._call_counter = 0

    @cached(request_cache)
    async def generate(self, department: str) -> str:
        self._call_counter += 1
        await asyncio.sleep(0.1)
        timestamp = time.time()
        return f"{department}:{self._call_counter}@{timestamp:.0f}"


async def main() -> None:
    generator = ReportGenerator()
    first = await generator.generate("sales")
    second = await generator.generate("sales")
    print(first == second)  # True -> cached result

    await asyncio.sleep(2.1)  # TTL expired
    third = await generator.generate("sales")
    print(first == third)  # False -> cache miss, recomputed


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

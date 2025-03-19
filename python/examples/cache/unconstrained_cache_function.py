import asyncio
import sys
import traceback

from beeai_framework.cache.unconstrained_cache import UnconstrainedCache
from beeai_framework.errors import FrameworkError


async def main() -> None:
    cache: UnconstrainedCache[int] = UnconstrainedCache()

    async def fibonacci(n: int) -> int:
        cache_key = str(n)
        cached = await cache.get(cache_key)
        if cached:
            return int(cached)

        if n < 1:
            result = 0
        elif n <= 2:
            result = 1
        else:
            result = await fibonacci(n - 1) + await fibonacci(n - 2)

        await cache.set(cache_key, result)
        return result

    print(await fibonacci(10))  # 55
    print(await fibonacci(9))  # 34 (retrieved from cache)
    print(f"Cache size {await cache.size()}")  # 10


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

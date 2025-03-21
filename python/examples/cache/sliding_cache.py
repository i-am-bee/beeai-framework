import asyncio
import sys
import traceback

from beeai_framework.cache.sliding_cache import SlidingCache
from beeai_framework.errors import FrameworkError


async def main() -> None:
    cache: SlidingCache[int] = SlidingCache(
        size=3,  # (required) number of items that can be live in the cache at a single moment
        ttl=1,  # // (optional, default is Infinity) Time in seconds after the element is removed from a cache
    )

    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)

    await cache.set("d", 4)  # overflow - cache internally removes the oldest entry (key "a")

    print(await cache.has("a"))  # False
    print(await cache.size())  # 3


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

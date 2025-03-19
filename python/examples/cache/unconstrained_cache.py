import asyncio
import sys
import traceback

from beeai_framework.cache.unconstrained_cache import UnconstrainedCache
from beeai_framework.errors import FrameworkError


async def main() -> None:
    cache: UnconstrainedCache[int] = UnconstrainedCache()

    await cache.set("a", 1)
    print(await cache.has("a"))  # True
    print(await cache.size())  # 1


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

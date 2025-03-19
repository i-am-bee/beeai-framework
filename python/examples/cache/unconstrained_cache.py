import asyncio
import sys
import traceback

from beeai_framework.cache.unconstrained_cache import UnconstrainedCache
from beeai_framework.errors import FrameworkError


async def main() -> None:
    cache: UnconstrainedCache[int] = UnconstrainedCache()

    # Save
    await cache.set("a", 1)
    await cache.set("b", 2)

    # Read
    result = await cache.has("a")
    print(result)  # True

    # Meta
    print(cache.enabled)  # True
    print(await cache.has("a"))  # True
    print(await cache.has("b"))  # True
    print(await cache.has("c"))  # False
    print(await cache.size())  # 2

    # Delete
    await cache.delete("a")
    print(await cache.has("a"))  # False

    # Clear
    await cache.clear()
    print(await cache.size())  # 0


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

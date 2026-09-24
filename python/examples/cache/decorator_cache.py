import asyncio
import sys
import time
import traceback

from beeai_framework.cache import SlidingCache, cached
from beeai_framework.errors import FrameworkError

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

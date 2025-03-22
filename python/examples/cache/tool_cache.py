import asyncio
import sys
import traceback

from beeai_framework.cache.sliding_cache import SlidingCache
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search.wikipedia import (
    WikipediaTool,
    WikipediaToolInput,
)


async def main() -> None:
    wikipedia_client = WikipediaTool({"full_text": True, "cache": SlidingCache(size=100, ttl=5 * 60)})

    print(await wikipedia_client.cache.size())  # 0
    tool_input = WikipediaToolInput(query="United States")
    first = await wikipedia_client.run(tool_input)
    print(await wikipedia_client.cache.size())  # 1

    # new request with the EXACTLY same input will be retrieved from the cache
    tool_input = WikipediaToolInput(query="United States")
    second = await wikipedia_client.run(tool_input)
    print(first.get_text_content() == second.get_text_content())  # True
    print(await wikipedia_client.cache.size())  # 1


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

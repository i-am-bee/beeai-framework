import asyncio
import sys
import traceback

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search.wikipedia import (
    WikipediaTool,
    WikipediaToolInput,
)


async def main() -> None:
    wikipedia_client = WikipediaTool({"full_text": True})
    tool_input = WikipediaToolInput(query="bee")
    result = await wikipedia_client.run(tool_input)
    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

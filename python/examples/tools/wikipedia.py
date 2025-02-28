import asyncio

from beeai_framework.tools.search.wikipedia import (
    WikipediaTool,
    WikipediaToolInput,
)


async def main() -> None:
    wikipedia_client = WikipediaTool({"full_text": True})
    input = WikipediaToolInput(query="bee")
    result = await wikipedia_client.run(input)
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())

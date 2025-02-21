import asyncio

from beeai_framework.tools.search.wikipedia import (
    WikipediaSearchTool,
    WikipediaSearchToolInput,
)


async def main() -> None:
    wikipedia_client = WikipediaSearchTool(full_text=True)
    input = WikipediaSearchToolInput(query="bee")
    result = wikipedia_client.run(input)
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())

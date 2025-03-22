import asyncio
import sys
import traceback

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.cache.sliding_cache import SlidingCache
from beeai_framework.errors import FrameworkError


async def main() -> None:
    llm = OllamaChatModel("llama3.1")
    llm.config(parameters=ChatModelParameters(max_tokens=25), cache=SlidingCache(size=50))

    print(await llm.cache.size())  # 0
    first = await llm.create(messages=[UserMessage("Who is Amilcar Cabral?")])
    print(await llm.cache.size())  # 1

    # new request with the EXACTLY same input will be retrieved from the cache
    second = await llm.create(messages=[UserMessage("Who is Amilcar Cabral?")])
    print(first.get_text_content() == second.get_text_content())  # True
    print(await llm.cache.size())  # 1


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

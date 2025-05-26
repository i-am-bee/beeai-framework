import asyncio
import sys
import traceback

from beeai_framework.backend import EmbeddingModel
from beeai_framework.errors import FrameworkError


async def main() -> None:
    embedding_llm = EmbeddingModel.from_name("ollama:nomic-embed-text")

    response = await embedding_llm.create(["Text", "to", "embed"])

    for row in response.embeddings:
        print(*row)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

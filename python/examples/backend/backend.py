import asyncio
import sys
import traceback

from beeai_framework.backend.backend import Backend
from beeai_framework.errors import FrameworkError


async def main() -> None:
    backend = Backend.from_provider("ollama")
    print(backend.chat.model_id)
    print(backend.embedding.model_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

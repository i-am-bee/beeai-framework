import asyncio
import json
import sys
import traceback

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


async def main() -> None:
    emitter: Emitter = Emitter(namespace=["base"])

    emitter.match("*.*", lambda data, event: print(f"Received event '{event.path}' with data {json.dumps(data)}"))

    await emitter.emit("start", {"id": 123})
    await emitter.emit("end", {"id": 123})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

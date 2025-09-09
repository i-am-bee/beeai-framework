import asyncio
import json
import sys
import traceback
from typing import Any

from beeai_framework.emitter import Emitter, EventMeta
from beeai_framework.errors import FrameworkError


async def main() -> None:
    # Get the root emitter or create your own
    root = Emitter.root()

    # Listen to all events that will get emitted
    @root.on("*.*")
    async def handle_new_event(data: Any, event: EventMeta) -> None:
        print(f"Received event '{event.name}' ({event.path}) with data {json.dumps(data)}")

    await root.emit("start", {"id": 123})
    await root.emit("end", {"id": 123})

    root.off(callback=handle_new_event)  # deregister a listener


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

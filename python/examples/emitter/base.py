# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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

    # Define a listener that prints an incoming event
    # handler can be async or sync function
    async def on_new_event(data: Any, event: EventMeta) -> None:
        print(f"Received event '{event.name}' ({event.path}) with data {json.dumps(data)}")

    # Setup a listener for all events on the root emitter
    # *.* -> match all events including those emitted in sub-emitters
    cleanup = root.match("*.*", on_new_event)

    await root.emit("start", {"id": 123})
    await root.emit("end", {"id": 123})

    cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

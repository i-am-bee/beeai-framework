import asyncio
import sys
import traceback
from typing import Any

from beeai_framework.emitter import Emitter, EventMeta
from beeai_framework.errors import FrameworkError


async def main() -> None:
    # Create an emitter
    emitter = Emitter.root().child(
        namespace=["bee", "demo"],
        creator={},  # typically a class
        context={},  # custom data (propagates to the event's context property)
        group_id=None,  # optional id for grouping common events (propagates to the event's groupId property)
    )

    @emitter.on()
    async def on_start(data: dict[str, Any], event: EventMeta) -> None:
        print(f"Received '{event.name}' event with id '{data['id']}'")

    # Listen for "update" event
    cleanup = emitter.on(
        "update", lambda data, event: print(f"Received '{event.name}' with id '{data['id']}' and data '{data['data']}'")
    )
    cleanup()  # deregister a listener

    # Listen for "success" event
    @emitter.on("success")
    async def custom_name(data: dict[str, Any], event: EventMeta) -> None:
        print(f"Received '{event.name}' event with the following data", data)

    await emitter.emit("start", {"id": 123})
    await emitter.emit("update", {"id": 123, "data": "Hello Bee!"})
    await emitter.emit("success", {"id": 123, "result": "Hello world!"})

    emitter.off("success", custom_name)  # deregister a listener


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

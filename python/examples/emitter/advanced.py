import asyncio
import sys
import traceback

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


async def main() -> None:
    # Create emitter with a type support
    emitter = Emitter.root().child(
        namespace=["bee", "demo"],
        creator={},  # typically a class
        context={},  # custom data (propagates to the event's context property)
        group_id=None,  # optional id for grouping common events (propagates to the event's groupId property)
        trace=None,  # data to identify what emitted what and in which context (internally used by framework components)
    )

    # Listen for "start" event
    emitter.on("start", lambda data, event: print(f"Received '{event.name}' event with id '{data['id']}'"))

    # Listen for "update" event
    emitter.on(
        "update", lambda data, event: print(f"Received '{event.name}' with id '{data['id']}' and data '{data['data']}'")
    )

    await emitter.emit("start", {"id": 123})
    await emitter.emit("update", {"id": 123, "data": "Hello Bee!"})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

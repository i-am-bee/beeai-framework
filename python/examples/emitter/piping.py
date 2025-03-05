import asyncio
import sys
import traceback

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


async def main() -> None:
    first: Emitter = Emitter(namespace=["app"])

    first.match(
        "*.*",
        lambda data, event: print(
            f"'first' has retrieved the following event '{event.path}', isDirect: {event.source == first}"
        ),
    )

    second: Emitter = Emitter(namespace=["app", "llm"])

    second.match(
        "*.*",
        lambda data, event: print(
            f"'second' has retrieved the following event '{event.path}', isDirect: {event.source == second}"
        ),
    )

    # Propagate all events from the 'second' emitter to the 'first' emitter
    unpipe = second.pipe(first)

    await first.emit("a", {})
    await second.emit("b", {})

    print("Unpipe")
    unpipe()

    await first.emit("c", {})
    await second.emit("d", {})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

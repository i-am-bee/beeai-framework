import asyncio
import re
import sys
import traceback

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


async def main() -> None:
    emitter = Emitter.root().child(namespace=["app"])
    model = OllamaChatModel()

    # Match events by a concrete name (strictly typed)
    emitter.on("update", lambda data, event: print(data, ": on update"))

    # Match all events emitted directly on the instance (not nested)
    emitter.on("*", lambda data, event: print(data, ": match all instance"))

    # Match all events (included nested)
    cleanup = Emitter.root().on("*.*", lambda data, event: print(data, ": match all nested"))

    # Match events by providing a filter function
    model.emitter.on(
        lambda event: isinstance(event.creator, ChatModel), lambda data, event: print(data, ": match ChatModel")
    )

    # Match events by regex
    emitter.on(re.compile(r"watsonx"), lambda data, event: print(data, ": match regex"))

    await emitter.emit("update", "update")
    await Emitter.root().emit("root", "root")
    await model.emitter.emit("model", "model")

    cleanup()  # You can remove a listener from an emitter by calling the cleanup function it returns


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

import asyncio
import sys
import traceback

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


async def main() -> None:
    emitter = Emitter.root().child(namespace=["app"])
    model = OllamaChatModel()

    # Match events by a concrete name (strictly typed)
    emitter.on("update", lambda data, event: print(data, ": on update"))

    # Match all events emitted directly on the instance (not nested)
    emitter.match("*", lambda data, event: print(data, ": match all instance"))

    # Match all events (included nested)
    Emitter.root().match("*.*", lambda data, event: print(data, ": match all nested"))

    # Match events by providing a filter function
    model.emitter.match(
        lambda event: isinstance(event.creator, ChatModel), lambda data, event: print(data, ": match ChatModel")
    )

    await emitter.emit("update", "update")
    await Emitter.root().emit("root", "root")
    await model.emitter.emit("model", "model")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

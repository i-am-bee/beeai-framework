import asyncio
import sys
import traceback

from beeai_framework import SystemMessage, UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader

reader = ConsoleReader()


async def main() -> None:
    memory = UnconstrainedMemory()
    await memory.add_many(
        [
            SystemMessage("Always respond very concisely."),
            UserMessage("Give me the first 5 prime numbers."),
        ]
    )

    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    response = (
        await llm.create(messages=memory.messages, stream=True)
        .observe(
            lambda emitter: emitter.on(
                "start",
                lambda data, event: print(
                    "On start", *(message.to_plain() for message in data.input.messages), sep="\n"
                ),
            )
        )
        .observe(
            lambda emitter: emitter.on(
                "new_token",
                lambda data, event: print(
                    "On new_token", *(message.to_plain() for message in data.value.messages), sep="\n"
                ),
            )
        )
        .observe(
            lambda emitter: emitter.on(
                "success",
                lambda data, event: print(
                    "On success", *(message.to_plain() for message in data.value.messages), sep="\n"
                ),
            )
        )
        .observe(lambda emitter: emitter.on("error", lambda data, event: print(event.name, data.error)))
        .observe(lambda emitter: emitter.on("finish", lambda data, event: print("On finish", data)))
    )

    print("Response:", response.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

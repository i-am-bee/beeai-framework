import asyncio
import sys
import traceback

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.backend import (
    ChatModelErrorEvent,
    ChatModelNewTokenEvent,
    ChatModelStartEvent,
    ChatModelSuccessEvent,
    UserMessage,
)
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from examples.helpers.io import ConsoleReader


async def main() -> None:
    llm = OllamaChatModel("llama3.1")
    llm.parameters.temperature = 1

    reader = ConsoleReader()

    for prompt in reader:

        def on_start(data: ChatModelStartEvent, meta: EventMeta) -> None:
            reader.write("LLM 🤖 (start)", str(data.input.model_dump()))

        def on_error(data: ChatModelErrorEvent, meta: EventMeta) -> None:
            reader.write("LLM 🤖 (error)", str(data.error.explain()))

        def on_new_token(data: ChatModelNewTokenEvent, meta: EventMeta) -> None:
            reader.write("LLM 🤖 (new_token)", data.value.get_text_content())

        def on_success(data: ChatModelSuccessEvent, meta: EventMeta) -> None:
            reader.write("LLM 🤖 (success)", data.value.get_text_content())
            if data.value.usage:
                reader.write("LLM 🤖 (usage)", str(data.value.usage.model_dump()))

        response = (
            await llm.create(messages=[UserMessage(prompt)], stream=True)
            .on("start", on_start)
            .on("error", on_error)
            .on("success", on_success)
            .on("new_token", on_new_token)
        )

        reader.write("ℹ️", f"Received {len(response.messages)} chunks.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

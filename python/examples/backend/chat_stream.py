import asyncio
import sys
import traceback

from beeai_framework import UserMessage
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.errors import FrameworkError
from examples.helpers.io import ConsoleReader


async def main() -> None:
    llm = OllamaChatModel("llama3.1")

    reader = ConsoleReader()

    for prompt in reader:
        response = await llm.create(messages=[UserMessage(prompt)]).observe(
            lambda emitter: emitter.match(
                "*", lambda data, event: reader.write(f"LLM 🤖 (event: {event.name})", str(data))
            )
        )

        reader.write("LLM 🤖 (txt) : ", response.get_text_content())
        reader.write("LLM 🤖 (raw) : ", "\n".join([str(msg.to_plain()) for msg in response.messages]))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

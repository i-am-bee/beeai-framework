import asyncio
import sys
import traceback

from beeai_framework import ReActAgent, UnconstrainedMemory
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.errors import FrameworkError


async def main() -> None:
    agent = ReActAgent(
        llm=OllamaChatModel("llama3.1"),
        memory=UnconstrainedMemory(),
        tools=[],
    )

    # Matching events on the instance level
    agent.emitter.match("*.*", lambda data, event: None)

    # Matching events on the execution (run) level
    await agent.run("Hello agent!").observe(
        lambda emitter: emitter.match("*.*", lambda data, event: print(f"RUN LOG: received event '{event.path}'"))
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

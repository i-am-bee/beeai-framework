import asyncio
import logging
import sys
import traceback

from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.utils import BeeLogger


async def main() -> None:
    logger = BeeLogger("app", level=logging.TRACE)

    agent = BeeAgent(llm=ChatModel.from_name("ollama:granite3.1-dense:8b"), tools=[], memory=UnconstrainedMemory())

    output: BeeRunOutput = await agent.run("Hello!").observe(
        lambda emitter: emitter.on(
            "update", lambda data, event: logger.info(f"Event {event.path} triggered by {type(event.creator).__name__}")
        )
    )

    logger.info(f"Agent 🤖 : {output.result.text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

import asyncio
import logging
import sys
import traceback

from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.utils import BeeLogger


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = BeeAgent(llm=llm, tools=[], memory=UnconstrainedMemory())
    logger = BeeLogger("app", level=logging.TRACE)

    def update_callback(data: dict, event: EventMeta) -> None:
        logger.info(f"Event {event.path} triggered by {event.creator.__class__.__name__}")
        logger.info(f"Agent({data['update']['key']} 🤖 : " + data["update"]["parsedValue"])

    def on_update(emitter: Emitter) -> None:
        emitter.on("update", update_callback)

    output: BeeRunOutput = await agent.run("Hello!").observe(on_update)

    logger.info(f"Agent 🤖 : {output.result.text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

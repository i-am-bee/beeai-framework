"""Expose a `LiteAgent` over Zed's Agent Client Protocol (stdio).

`LiteAgent` is not one of the three pre-registered agent types (`RequirementAgent`,
`ToolCallingAgent`, `ReActAgent`) because it has a different streaming surface — a
single-shot `run()` that emits `final_answer` chunks through its emitter instead of
an async iterator yielding memory updates. This example doubles as a template for
adapting any custom BeeAI agent: register a factory that builds an `ACPZedServerAgent`
with your own `run_turn` closure, and the rest of the protocol plumbing (session
lifecycle, cancellation, FS bridge) is reused unchanged.

Launch from Zed (`~/.config/zed/settings.json`):

    {
      "agent_servers": {
        "beeai-lite": {
          "command": "python",
          "args": ["/absolute/path/to/examples/serve/acp_zed_lite.py"]
        }
      }
    }

Prereqs: `pip install "beeai-framework[acp-zed,search]"` and a running Ollama with
the `granite4:micro` model pulled.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from acp import text_block, update_agent_message
from acp.interfaces import Client

from beeai_framework.adapters.acp_zed import (
    ACPZedReadFileTool,
    ACPZedServer,
    ACPZedServerConfig,
    ACPZedWriteFileTool,
)
from beeai_framework.adapters.acp_zed.serve._utils import PromptBlock, acp_zed_prompt_to_framework_msgs
from beeai_framework.adapters.acp_zed.serve.agent import ACPZedServerAgent
from beeai_framework.agents.lite import LiteAgent
from beeai_framework.backend import ChatModel, ChatModelOutput, ChatModelParameters, SystemMessage
from beeai_framework.emitter import EventMeta
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


def _lite_agent_factory(agent: LiteAgent, *, server: ACPZedServer[Any]) -> ACPZedServerAgent:
    """Adapter factory for `LiteAgent`.

    `LiteAgent.run()` streams text through an emitter `final_answer` event rather
    than through an async iterator of state snapshots. We bridge the synchronous
    emitter callback to the async ACP connection via a queue, so chunks flush to
    Zed as they arrive.
    """

    async def run_turn(session_id: str, prompt: list[PromptBlock], conn: Client, session: LiteAgent) -> str:
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def on_final_answer(data: ChatModelOutput, meta: EventMeta) -> None:
            chunk = data.get_text_content()
            if chunk:
                queue.put_nowait(chunk)

        async def consume() -> None:
            while (chunk := await queue.get()) is not None:
                await conn.session_update(session_id=session_id, update=update_agent_message(text_block(chunk)))

        cleanup = session.emitter.on("final_answer", on_final_answer)
        consumer = asyncio.create_task(consume())
        try:
            await session.run(acp_zed_prompt_to_framework_msgs(prompt))
        finally:
            cleanup()
            queue.put_nowait(None)
            await consumer
        return "end_turn"

    return server._build_wrapper(agent, run_turn)


async def _build_agent(server: ACPZedServer[Any]) -> LiteAgent:
    agent = LiteAgent(
        llm=ChatModel.from_name("ollama:granite4:micro", ChatModelParameters(stream=True)),
        tools=[
            ThinkTool(),
            OpenMeteoTool(),
            DuckDuckGoSearchTool(),
            ACPZedReadFileTool(server),
            ACPZedWriteFileTool(server),
        ],
    )
    await agent.memory.add(SystemMessage("You are a helpful coding and research assistant."))
    return agent


def main() -> None:
    # Register the custom factory before `serve()` inspects `_factories`.
    ACPZedServer.register_factory(LiteAgent, _lite_agent_factory, override=True)  # type: ignore[arg-type]

    server = ACPZedServer(
        config=ACPZedServerConfig(
            agent_name="beeai-lite",
            agent_description="BeeAI LiteAgent with web-search, weather, and think tools — plus workspace FS.",
        )
    )
    agent = asyncio.run(_build_agent(server))
    server.register(agent).serve()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

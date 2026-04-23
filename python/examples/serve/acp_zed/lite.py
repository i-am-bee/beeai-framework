"""Expose a `LiteAgent` over Zed's Agent Client Protocol (stdio).

`LiteAgent` is a pre-registered agent type in the adapter alongside
`RequirementAgent`, `ToolCallingAgent`, and `ReActAgent` — just register and serve.

Launch from Zed (`~/.config/zed/settings.json`):

    {
      "agent_servers": {
        "beeai-lite": {
          "command": "python",
          "args": ["/absolute/path/to/examples/serve/acp_zed/lite.py"]
        }
      }
    }

Prereqs: `pip install "beeai-framework[acp-zed,search]"` and a running Ollama with
the `granite4:micro` model pulled.
"""

from __future__ import annotations

import asyncio
import sys

from beeai_framework.adapters.acp_zed import (
    ACPZedReadFileTool,
    ACPZedServer,
    ACPZedServerConfig,
    ACPZedWriteFileTool,
)
from beeai_framework.agents.lite import LiteAgent
from beeai_framework.backend import ChatModel, ChatModelParameters, SystemMessage
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


async def _build_agent(server: ACPZedServer[LiteAgent]) -> LiteAgent:
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
    server: ACPZedServer[LiteAgent] = ACPZedServer(
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

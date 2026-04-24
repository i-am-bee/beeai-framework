"""Expose a `LiteAgent` over Zed's Agent Client Protocol (stdio).

Uses only generic BeeAI tools — the same agent definition runs unchanged over any
other serve mode (or in a plain script). The ACP Zed adapter installs protocol-
aware shell/file backends for the turn, so `ShellTool` routes through the editor's
terminal widget, `FileReadTool`/`FileEditTool` go through `fs/*_text_file`, and
`io_confirm` becomes `session/request_permission`.

Launch from Zed (`~/.config/zed/settings.json`):

    {
      "agent_servers": {
        "beeai-lite": {
          "command": "python",
          "args": ["/absolute/path/to/examples/serve/acp_zed/lite.py"]
        }
      }
    }

Prereqs: `pip install "beeai-framework[acp-zed,search]"`.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from beeai_framework.adapters.acp_zed import ACPZedServer, ACPZedServerConfig
from beeai_framework.agents.lite import LiteAgent
from beeai_framework.backend import ChatModel, ChatModelParameters, SystemMessage  # noqa: F401
from beeai_framework.tools.code import ShellTool
from beeai_framework.tools.filesystem import FileEditTool, FileReadTool, GlobTool, GrepTool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool  # noqa: F401
from beeai_framework.tools.think import ThinkTool  # noqa: F401
from beeai_framework.tools.weather import OpenMeteoTool  # noqa: F401

load_dotenv()


async def _build_agent() -> LiteAgent:
    agent = LiteAgent(
        llm=ChatModel.from_name("openai:gpt-5.4-mini", ChatModelParameters(stream=True)),
        tools=[
            # ThinkTool(),
            # OpenMeteoTool(),
            # DuckDuckGoSearchTool(),
            FileReadTool(),
            FileEditTool(),
            ShellTool(),
            GlobTool(),
            GrepTool(),
        ],
    )
    # await agent.memory.add(SystemMessage("You are a helpful coding and research assistant."))
    return agent


def main() -> None:
    server: ACPZedServer[LiteAgent] = ACPZedServer(
        config=ACPZedServerConfig(
            agent_name="beeai-lite",
            agent_description="BeeAI LiteAgent",
        )
    )
    agent = asyncio.run(_build_agent())
    server.register(agent).serve()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

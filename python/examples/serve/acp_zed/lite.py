"""Full coding-tool loadout on `LiteAgent`, exposed over Zed ACP (stdio).

The same tools you'd give any BeeAI agent — no ACP subclasses. The adapter's
per-turn context installs ACP-routed backends so `ShellTool` routes through
the editor's terminal widget, `FileReadTool`/`FileEditTool` go through
`fs/*_text_file`, and `io_confirm` becomes `session/request_permission`.

Extra tools like search/think/weather are one import away — see
`beeai_framework.tools.search`, `.think`, `.weather`.

Launch from Zed (`~/.config/zed/settings.json`):

    {
      "agent_servers": {
        "beeai-lite": {
          "command": "python",
          "args": ["/absolute/path/to/examples/serve/acp_zed/lite.py"]
        }
      }
    }

Prereqs: `pip install "beeai-framework[acp-zed]"` plus credentials for
whichever backend you wire up (OpenAI / Ollama / …).
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from beeai_framework.adapters.acp_zed import ACPZedServer, ACPZedServerConfig
from beeai_framework.agents.lite import LiteAgent
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.code import ShellTool
from beeai_framework.tools.filesystem import FileEditTool, FileReadTool, GlobTool, GrepTool

load_dotenv()


async def _build_agent() -> LiteAgent:
    return LiteAgent(
        llm=ChatModel.from_name("openai:gpt-5.4-mini", ChatModelParameters(stream=True)),
        tools=[
            FileReadTool(),
            FileEditTool(),
            ShellTool(),
            GlobTool(),
            GrepTool(),
        ],
    )


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

"""Smallest viable coding agent over Zed's Agent Client Protocol (stdio).

Three generic tools — `FileReadTool`, `FileEditTool`, `ShellTool` — and nothing
ACP-specific in the agent definition. When served through `ACPZedServer`, the
adapter installs ACP-routed backends for the turn, so file reads/writes flow
through `fs/read_text_file` / `fs/write_text_file` and shell commands open in
Zed's terminal widget. The same agent definition would run unchanged as a CLI
script or over any other serve mode.

Launch from Zed (`~/.config/zed/settings.json`):

    {
      "agent_servers": {
        "beeai": {
          "command": "python",
          "args": ["/absolute/path/to/examples/serve/acp_zed/simple.py"]
        }
      }
    }

Prereqs: `pip install "beeai-framework[acp-zed]"`, Ollama running with
`granite4:micro`.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from beeai_framework.adapters.acp_zed import ACPZedServer
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.code import ShellTool
from beeai_framework.tools.filesystem import FileEditTool, FileReadTool

load_dotenv()


async def _build_agent() -> RequirementAgent:
    return RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=[FileReadTool(), FileEditTool(), ShellTool()],
        memory=UnconstrainedMemory(),
        name="beeai-coder",
        description="A BeeAI RequirementAgent running as a Zed ACP agent.",
    )


def main() -> None:
    agent = asyncio.run(_build_agent())
    ACPZedServer().register(agent).serve()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

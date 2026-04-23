"""Expose a BeeAI RequirementAgent as a Zed Agent Client Protocol program (stdio).

Launch from Zed (`~/.config/zed/settings.json`):

    {
      "agent_servers": {
        "beeai": {
          "command": "python",
          "args": ["/absolute/path/to/examples/serve/acp_zed.py"]
        }
      }
    }

Prereqs: `pip install "beeai-framework[acp-zed]"`.
"""

from beeai_framework.adapters.acp_zed import (
    ACPZedReadFileTool,
    ACPZedServer,
    ACPZedWriteFileTool,
)
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory


def main() -> None:
    server = ACPZedServer()
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=[ACPZedReadFileTool(server), ACPZedWriteFileTool(server)],
        memory=UnconstrainedMemory(),
        name="beeai-coder",
        description="A BeeAI RequirementAgent running as a Zed ACP agent.",
    )
    server.register(agent).serve()


if __name__ == "__main__":
    main()

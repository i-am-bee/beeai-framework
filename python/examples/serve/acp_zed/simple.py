"""Expose a BeeAI RequirementAgent as a Zed Agent Client Protocol program (stdio).

Uses only generic BeeAI tools (`FileReadTool`, `FileEditTool`) — the adapter
installs ACP-routed backends for the turn, so reads + writes automatically flow
through `fs/read_text_file` / `fs/write_text_file` while the agent definition
stays protocol-agnostic.

Launch from Zed (`~/.config/zed/settings.json`):

    {
      "agent_servers": {
        "beeai": {
          "command": "python",
          "args": ["/absolute/path/to/examples/serve/acp_zed/simple.py"]
        }
      }
    }

Prereqs: `pip install "beeai-framework[acp-zed]"`.
"""

from beeai_framework.adapters.acp_zed import ACPZedServer
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.filesystem import FileEditTool, FileReadTool


def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=[FileReadTool(), FileEditTool()],
        memory=UnconstrainedMemory(),
        name="beeai-coder",
        description="A BeeAI RequirementAgent running as a Zed ACP agent.",
    )
    ACPZedServer().register(agent).serve()


if __name__ == "__main__":
    main()

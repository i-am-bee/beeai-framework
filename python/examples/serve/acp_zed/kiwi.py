"""Expose a Kiwi.com flight-search agent over Zed's Agent Client Protocol (stdio).

Wraps the `RequirementAgent` from `examples/playground/tests/kiwi.py` — it pulls
its toolset from the public Kiwi.com MCP server and answers flight-booking
questions inside Zed's agent panel. This is a domain-specific showcase; for
coding-focused examples that use the adapter's auto-routed shell + file
backends, see `simple.py` and `lite.py`.

Launch from Zed (`~/.config/zed/settings.json`):

    {
      "agent_servers": {
        "beeai-kiwi": {
          "command": "python",
          "args": ["/absolute/path/to/examples/serve/acp_zed/kiwi.py"]
        }
      }
    }

Prereqs: `pip install "beeai-framework[acp-zed,mcp]"` plus whatever secrets
`watsonx` or your chosen backend needs in `.env`.
"""

import asyncio
import sys

from dotenv import load_dotenv
from mcp.client.streamable_http import streamablehttp_client

from beeai_framework.adapters.acp_zed import ACPZedServer, ACPZedServerConfig
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.mcp import MCPTool

load_dotenv()


async def _build_agent() -> RequirementAgent:
    mcp_tools = await MCPTool.from_client(streamablehttp_client("https://mcp.kiwi.com"))
    return RequirementAgent(
        llm=ChatModel.from_name("watsonx:ibm/granite-4-h-small", ChatModelParameters(temperature=0)),
        tools=[*mcp_tools],
        name="beeai-kiwi",
        description="Finds and prices flights via the Kiwi.com MCP server.",
    )


def main() -> None:
    agent = asyncio.run(_build_agent())
    ACPZedServer(
        config=ACPZedServerConfig(
            agent_name="beeai-kiwi",
            agent_description="BeeAI flight-search agent backed by Kiwi.com MCP tools.",
        )
    ).register(agent).serve()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

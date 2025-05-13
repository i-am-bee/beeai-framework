import asyncio
from typing import Any

from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.agents.tool_calling.abilities import ConditionalAbility
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import tool
from beeai_framework.tools.weather import OpenMeteoTool


@tool
def current_user() -> dict[str, Any]:
    """Retrieves information about the current user."""

    return {"name": "Alex", "address": "London, United Kingdom"}


async def main() -> None:
    agent = ToolCallingAgent(
        llm=ChatModel.from_name("ollama:llama3.1"),
        memory=UnconstrainedMemory(),
        abilities=[
            ConditionalAbility(current_user, force_at_step=0),
            ConditionalAbility(OpenMeteoTool(), force_at_step=1),
        ],
    )

    result = await agent.run(prompt="What is the current weather?")
    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())

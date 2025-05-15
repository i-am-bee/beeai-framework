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


@tool
def send_email(content: str) -> None:
    """Sends a summary e-mail to the author"""

    return None


async def main() -> None:
    agent = ToolCallingAgent(
        llm=ChatModel.from_name("ollama:qwen2.5:1.5b"),
        memory=UnconstrainedMemory(),
        abilities=[
            ConditionalAbility(current_user, force_at_step=0, max_invocations=1),
            ConditionalAbility(OpenMeteoTool(), min_invocations=1, can_be_used_in_row=False),
            ConditionalAbility(send_email, only_after="OpenMeteoTool"),
        ],
    )

    result = await agent.run(prompt="What is the current weather?")
    print(result.result.text)
    # print(json.loads(result.result.text)['location'])


if __name__ == "__main__":
    asyncio.run(main())

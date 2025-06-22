import asyncio
from typing import Any

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement  # noqa: F401
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool, tool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


@tool
def get_user() -> dict[str, Any]:
    """Retrieve information about the current user."""
    return {"name": "John", "location": "Boston"}


async def main() -> None:
    """
    🐛 **Problem:**
    The agent does not use the user's location when answering (run the script to observe this).

    💡 **How to Fix:**
    Uncomment one of the solution lines in the `requirements` list to ensure the agent
    retrieves user information before answering weather-related queries.

    - Solution 1: Force the agent to call `get_user` at the very beginning.
    - Solution 2: Ensure the agent only calls `OpenMeteoTool` after obtaining user info.
    """

    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = RequirementAgent(
        llm=llm,
        tools=[OpenMeteoTool(), ThinkTool(), get_user],
        # Log all tool calls to the console for easier debugging
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            # 💡 Solution 1: Force the agent to call 'get_user' at the start
            # ConditionalRequirement(target=get_user, force_at_step=1),
            # 💡 Solution 2: Ensure 'OpenMeteoTool' is only called after 'get_user'
            # ConditionalRequirement(target=OpenMeteoTool, only_after=[get_user]),
        ],
    )

    response = await agent.run("What's the current weather?")
    print(response.answer.text)


if __name__ == "__main__":
    asyncio.run(main())

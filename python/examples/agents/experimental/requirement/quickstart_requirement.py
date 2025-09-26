import asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


# Create an agent that plans activities based on weather and events
async def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3"),
        tools=[
            ThinkTool(),  # to reason
            OpenMeteoTool(),  # retrieve weather data
            DuckDuckGoSearchTool(),  # search web
        ],
        instructions="Plan activities for a given destination based on current weather and events.",
        requirements=[
            # Force thinking first
            ConditionalRequirement(ThinkTool, force_at_step=1),
            # Search only after getting weather and at least once
            ConditionalRequirement(DuckDuckGoSearchTool, only_after=[OpenMeteoTool], min_invocations=1),
            # Weather tool be used at least once but not consecutively
            ConditionalRequirement(OpenMeteoTool, consecutive_allowed=False, min_invocations=1),
        ],
    )
    # Run with execution logging
    response = await agent.run("What to do in Boston?").middleware(GlobalTrajectoryMiddleware())
    print(f"Final Answer: {response.last_message.text}")


if __name__ == "__main__":
    asyncio.run(main())

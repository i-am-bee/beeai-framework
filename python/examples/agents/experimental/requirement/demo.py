import asyncio

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


async def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[ThinkTool(), OpenMeteoTool(), DuckDuckGoSearchTool()],
        instructions="Plan activities for a given destination based on current weather and events.",
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1, max_invocations=3),
            ConditionalRequirement(
                DuckDuckGoSearchTool, only_after=[OpenMeteoTool], min_invocations=1, max_invocations=2
            ),
        ],
    )

    response = await agent.run("What to do in Boston?").middleware(GlobalTrajectoryMiddleware(excluded=[]))
    print(response.answer.text)


if __name__ == "__main__":
    asyncio.run(main())

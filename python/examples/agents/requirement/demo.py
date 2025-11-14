import asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


async def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name(
            "groq:llama-3.1-8b-instant",
            ChatModelParameters(stream=True),
            fallback_failed_generation=False,
        ),
        tools=[ThinkTool(), OpenMeteoTool(), DuckDuckGoSearchTool()],
        instructions="Plan activities for a given destination based on current weather and events.",
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1, max_invocations=3),
            ConditionalRequirement(
                DuckDuckGoSearchTool,
                only_after=[OpenMeteoTool],
                min_invocations=1,
                max_invocations=2,
            ),
        ],
    )

    response = await agent.run(
        "What to do in Boston today? Use DuckDuckGo tool now.", max_retries_per_step=3, total_max_retries=10
    ).middleware(GlobalTrajectoryMiddleware(included=[Tool]))
    print(response.last_message.text)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


async def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=[ThinkTool(), WikipediaTool(), OpenMeteoTool()],
        requirements=[ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False)],
    )

    response = await agent.run("What to do in Boston today?").middleware(GlobalTrajectoryMiddleware(included=[Tool]))
    print(response.last_message.text)


if __name__ == "__main__":
    asyncio.run(main())

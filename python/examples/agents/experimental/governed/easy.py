import asyncio

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements import Requirement
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import tool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool


@tool
def send_summary(content: str) -> None:
    pass


async def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[ThinkTool(), DuckDuckGoSearchTool(), send_summary],
        requirements=[
            ConditionalRequirement(send_summary, min_invocations=1, max_invocations=1),
        ],
    )
    response = await agent.run("What is the capital of France?").middleware(
        GlobalTrajectoryMiddleware(excluded=[Requirement])
    )
    print(response.result.text)


if __name__ == "__main__":
    asyncio.run(main())

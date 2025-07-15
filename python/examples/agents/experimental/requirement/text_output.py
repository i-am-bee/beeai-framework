import asyncio

from dotenv import load_dotenv

from beeai_framework.agents import AgentContext
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware

load_dotenv()


async def main() -> None:
    agent = RequirementAgent(llm=ChatModel.from_name("ollama:granite3.3:8b"))

    response = await agent.run(
        # pass the task
        "Write a step-by-step tutorial on how to bake bread.",
        # nudge the model to format an output
        context=AgentContext(
            expected_output="The output should be an ordered list of steps. Each step should be ideally one sentence."
        ),
    ).middleware(GlobalTrajectoryMiddleware())  # log intermediate steps

    print(response.answer.text)


if __name__ == "__main__":
    asyncio.run(main())

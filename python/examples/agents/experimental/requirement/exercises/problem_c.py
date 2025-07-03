# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from dotenv import load_dotenv

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement  # noqa: F401
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.think import ThinkTool

load_dotenv()


async def main() -> None:
    """
    🐛 **Problem:**
    The model enters a loop (ThinkTool is called three times) and then provides an incorrect answer.

    💡 **How to Fix:**
    Use `ConditionalRequirement` to:
      - Force the agent to call `ThinkTool` at the very beginning (step 1).
      - Limit the number of `ThinkTool` invocations to 1, preventing unnecessary cycles.
    """

    agent = RequirementAgent(
        llm=ChatModel.from_name("watsonx:meta-llama/llama-3-2-11b-vision-instruct"),
        tools=[ThinkTool()],
        requirements=[
            # 💡 Force 'ThinkTool' to be called at step 1, and only once
            # ConditionalRequirement(ThinkTool, force_at_step=1, max_invocations=1),
            # or
            # ConditionalRequirement(ThinkTool, force_at_step=1, consecutive_allowed=False),
        ],
        # Log all tool calls to the console for easier debugging
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    response = await agent.run(
        "A farmer has 10 cows, 5 chickens, and 2 horses. "
        "If we count all the animals' legs together, how many legs are there in total?"
    )
    print(response.answer.text)


if __name__ == "__main__":
    asyncio.run(main())

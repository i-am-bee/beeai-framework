# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement  # noqa: F401
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool, tool
from beeai_framework.tools.search.wikipedia import WikipediaTool


@tool
def send_summary(content: str) -> None:
    """
    Send a summarized answer to the user's email.
    """
    pass


async def main() -> None:
    """
    🐛 **Problem:**
    The agent does not call the `send_summary` tool before finishing its response.

    💡 **How to Fix:**
    Use `ConditionalRequirement` to ensure:
      - The agent calls `send_summary` exactly once.
      - The agent only produces the final answer after calling `send_summary`.
    """

    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = RequirementAgent(
        llm=llm,
        instructions=("After you resolve the task, send a summary to the user's email."),
        tools=[send_summary, WikipediaTool()],
        requirements=[
            # 💡 Uncomment following lines to fix the problem
            # -> Ensure 'send_summary' is called exactly once
            # -> Only allow 'final_answer' after 'send_summary' has been called
            # ConditionalRequirement(send_summary, min_invocations=1, max_invocations=1),
            # ConditionalRequirement("final_answer", force_after=send_summary),
        ],
        # Log all tool calls to the console for easier debugging
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    response = await agent.run("Tell me something about Greek history.")
    print(response.answer.text)


if __name__ == "__main__":
    asyncio.run(main())

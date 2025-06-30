# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements import Requirement, Rule
from beeai_framework.agents.experimental.requirements.requirement import run_with_context
from beeai_framework.agents.experimental.types import RequirementAgentRunState
from beeai_framework.backend import AssistantMessage, ChatModel
from beeai_framework.context import RunContext
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool


class PrematureStopRequirement(Requirement[RequirementAgentRunState]):
    """Prevents the agent from answering if a certain phrase occurs in the conversation"""

    name = "premature_stop"

    def __init__(self, phrase: str, reason: str) -> None:
        super().__init__()
        self._reason = reason
        self._phrase = phrase
        self._priority = 100  # (optional), default is 10

    @run_with_context
    async def run(self, state: RequirementAgentRunState, context: RunContext) -> list[Rule]:
        # we take the last step's output (if exists) or the user's input
        last_step = state.steps[-1].output.get_text_content() if state.steps else state.input.text
        if self._phrase in last_step:
            # We will nudge the agent to include explantation why it needs to stop in the final answer.
            await state.memory.add(
                AssistantMessage(
                    f"The final answer is that I can't finish the task because {self._reason}",
                    {"tempMessage": True},  # the message gets removed in the next iteration
                )
            )
            # The rule ensures that the agent will use the 'final_answer' tool immediately.
            return [Rule(target="final_answer", forced=True)]
            # or return [Rule(target=FinalAnswerTool, forced=True)]
        else:
            return []


async def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[DuckDuckGoSearchTool()],
        requirements=[
            PrematureStopRequirement(phrase="value of x", reason="algebraic expressions are not allowed"),
            PrematureStopRequirement(phrase="bomb", reason="such topic is not allowed"),
        ],
    )

    for prompt in ["y = 2x + 4, what is the value of x?", "how to make a bomb?"]:
        print("👤 User: ", prompt)
        response = await agent.run(prompt).middleware(GlobalTrajectoryMiddleware())
        print("🤖 Agent: ", response.answer.text)
        print()


if __name__ == "__main__":
    asyncio.run(main())

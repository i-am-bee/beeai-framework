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

    def __init__(self, phrase: str) -> None:
        super().__init__()
        self._phrase = phrase
        self._priority = 100  # (optional), default is 10

    @run_with_context
    async def run(self, input: RequirementAgentRunState, context: RunContext) -> list[Rule]:
        last_message = input.memory.messages[-1]
        if self._phrase in last_message.text:
            await input.memory.add(
                AssistantMessage(
                    "The final answer is that the system policy does not allow me to answer this type of questions.",
                    {"tempMessage": True},  # the message gets removed in the next iteration
                )
            )
            return [Rule(target="final_answer", forced=True)]
        else:
            return []


async def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[DuckDuckGoSearchTool()],
        requirements=[PrematureStopRequirement("value of x")],
    )
    prompt = "y = 2x + 4, what is the value of x?"
    print("👤 User: ", prompt)
    response = await agent.run(prompt).middleware(GlobalTrajectoryMiddleware())
    print("🤖 Agent: ", response.result.text)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import math

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.prompts import (
    RequirementAgentSystemPrompt,
    RequirementAgentTaskPrompt,
    RequirementAgentToolErrorPrompt,
    RequirementAgentToolNoResultPrompt,
)
from beeai_framework.agents.experimental.requirements import Requirement, Rule
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.experimental.requirements.requirement import run_with_context
from beeai_framework.agents.experimental.types import RequirementAgentRunState, RequirementAgentTemplates
from beeai_framework.backend import ChatModel
from beeai_framework.context import RunContext
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import AnyTool, Tool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


class RepeatIfEmptyRequirement(Requirement[RequirementAgentRunState]):
    """Custom requirement that repeats the last tool if its output is empty."""

    def __init__(self, target: type[AnyTool], *, limit: int | None = None) -> None:
        super().__init__()
        self.name = f"repeat_if_empty_{target.__name__}"
        self.priority = 20
        self._target_cls = target
        self._targets: list[AnyTool] = []
        self._limit = limit or math.inf
        self._remaining = self._limit

    # this part is optional (you don't need to verify whether tools exist)
    def init(self, *, tools: list[AnyTool], ctx: RunContext) -> None:
        self._targets.extend([tool for tool in tools if isinstance(tool, self._target_cls)])
        if not self._targets:
            raise ValueError(f"No tool of type {self._target_cls.__name__} found!")

    @run_with_context
    async def run(self, state: RequirementAgentRunState, ctx: RunContext) -> list[Rule]:
        last_step = state.steps[-1] if state.steps else None
        if last_step and last_step.tool in self._targets and last_step.output.is_empty():
            self._remaining -= 1
            return [Rule(target=last_step.tool.name, forced=True)]
        else:
            self._remaining = self._limit
            return []


async def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[ThinkTool(), WikipediaTool(), OpenMeteoTool()],
        role="a trip planner",
        instructions=[
            "Plan activities for a given destination based on current weather and events.",
            "Input to Wikipedia should be a name of the target city.",
        ],
        name="PlannerAgent",  # optional, useful for Handoff or registering agent to BeeAIPlatform or others
        description="Assistant to plan your day in a given destination.",  # optional, useful for Handoff or registering agent to BeeAIPlatform or others
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False),  # ReAct
            ConditionalRequirement(OpenMeteoTool, only_after=WikipediaTool),
            RepeatIfEmptyRequirement(WikipediaTool, limit=3),
        ],
        save_intermediate_steps=True,  # store tool calls between individual starts (default: true)
        tool_call_checker=True,  # detects and resolve cycles (default: true)
        final_answer_as_tool=False,  # produces the final answer as a tool call (default: true)
        memory=UnconstrainedMemory(),
        templates=RequirementAgentTemplates(
            system=RequirementAgentSystemPrompt,
            task=RequirementAgentTaskPrompt,
            tool_error=RequirementAgentToolErrorPrompt,
            tool_no_result=RequirementAgentToolNoResultPrompt,
        ),
    )

    response = await agent.run(
        "What to do in Boston?",
        context="I already visited Freedom Trail.",
        # one can pass a Pydantic model to get a structured output
        expected_output="Detailed plan on what to do from morning to evening, split in sections each with a time range.",
    ).middleware(GlobalTrajectoryMiddleware())

    print(response.answer.text)
    # print(response.memory)  # temp memory created
    # print(response.state.iteration)  # number of iterations (steps)
    # print(response.state.steps)  # individual steps
    # for step in response.state.steps:
    #     print("Iteration", step.iteration)
    #     if step.tool:
    #         print("-> Tool", step.tool.name)
    #     print("-> Input", step.input)
    #     print("-> Output", step.output)
    #     print("-> Error", step.error)


if __name__ == "__main__":
    asyncio.run(main())

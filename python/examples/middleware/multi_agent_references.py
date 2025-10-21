import asyncio

from beeai_framework.agents import AgentOutput, BaseAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import MessageTextContent
from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import RunContext, RunContextStartEvent, RunContextSuccessEvent, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.memory import BaseMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import StringToolOutput, Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool


class ReferenceResultMiddleware(RunMiddlewareProtocol):
    """
    Middleware for managing reference results in agent responses.

    Handle intermediate reference results in a multi-agent system. It captures the results of handoff tools and replaces
    placeholder variables in agent responses with actual output values retrieved from a store.
    The process includes binding the middleware to events, updating stored values, and ensuring
    placeholders are accurately replaced in messages.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}  # store intermediate results
        self._last_id: int = 0

    def bind(self, run: RunContext) -> None:
        agent = run.instance
        assert isinstance(agent, BaseAgent), "Input must be an agent"

        # Applies middleware to all handoff tools
        for tool in agent.meta.tools:
            if isinstance(tool, HandoffTool):
                run.emitter.on(
                    create_internal_event_matcher("start", tool),
                    self._handle_handoff_start,
                    EmitterOptions(match_nested=True, is_blocking=True, priority=1),
                )
                run.emitter.on(
                    create_internal_event_matcher("success", tool),
                    self._handle_handoff_success,
                    EmitterOptions(match_nested=True, is_blocking=True, priority=1),
                )

        # Replaces variables in the final answer
        run.emitter.on(
            create_internal_event_matcher("success", agent),
            self._handle_final_answer,
            EmitterOptions(match_nested=True, is_blocking=True, priority=1),
        )

    async def _handle_handoff_start(self, data: RunContextStartEvent, meta: EventMeta) -> None:
        """Overrides references in the last message from the real values from the store."""

        memory: BaseMemory = meta.context["state"]["memory"]
        last_message = memory.messages[-1] if memory.messages else None

        if last_message:
            self._update_message_with_store(last_message)

    async def _handle_handoff_success(self, data: RunContextSuccessEvent, meta: EventMeta) -> None:
        """Store the output of the handoff tool in the store and overrides the result to be a variable like $result1."""

        output: StringToolOutput = data.output
        self._last_id += 1
        self._store[str(self._last_id)] = output.get_text_content()
        output.result = self._format_result_id(self._last_id)

    async def _handle_final_answer(self, data: RunContextSuccessEvent, meta: EventMeta) -> None:
        """Replaces references in the final answer with the values from the store."""

        result: AgentOutput = data.output
        self._update_message_with_store(result.last_message)

    def _update_message_with_store(self, message: AnyMessage) -> None:
        """Replaces references in the message with the values from the store."""

        for chunk in message.content:
            if isinstance(chunk, MessageTextContent):
                for key, value in self._store.items():
                    chunk.text = chunk.text.replace(self._format_result_id(key), value)

    def _format_result_id(self, id: int | str) -> str:
        return f"$result{id}"


async def main() -> None:
    agent_b = RequirementAgent(
        llm="ollama:granite3.3", name="WeatherAgent", description="Weather agent", tools=[OpenMeteoTool()]
    )
    agent_c = RequirementAgent(
        llm="ollama:granite3.3", name="ResearcherAgent", description="Research agent", tools=[WikipediaTool()]
    )
    agent_a = RequirementAgent(
        llm="ollama:granite3.3",
        tools=[HandoffTool(agent_b), HandoffTool(agent_c)],
        instructions="You are a manager agent. "
        + "Results from managed agent will be stored as variables like $result1, $result2, $result3 and so on. "
        + "Treat them as plain strings.",
    )

    middleware = ReferenceResultMiddleware()
    response = (
        await agent_a.run("What's the current weather in Prague?")
        .middleware(middleware)
        .middleware(GlobalTrajectoryMiddleware(included=[Tool]))
    )
    print(response.last_message.text)


if __name__ == "__main__":
    asyncio.run(main())

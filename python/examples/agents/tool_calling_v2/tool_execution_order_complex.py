import asyncio

from beeai_framework.agents.tool_calling import ToolCallingAgent, ToolCallingAgentStartEvent
from beeai_framework.agents.tool_calling.abilities import ConditionalAbility, ReasoningAbility
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EventMeta
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool


async def main() -> None:
    agent = ToolCallingAgent(
        llm=ChatModel.from_name("ollama:llama3.1"),
        memory=UnconstrainedMemory(),
        abilities=[
            ConditionalAbility(ReasoningAbility(force=False), force_at_step=0, can_be_used_in_row=False),
            ConditionalAbility(WikipediaTool(), min_invocations=1, max_invocations=1),
            ConditionalAbility(OpenMeteoTool(), only_after="Wikipedia", min_invocations=1, max_invocations=1),
        ],
    )

    def on_start_request(data: ToolCallingAgentStartEvent, event: EventMeta) -> None:
        print("")
        print("=" * 32, "Step", data.state.iteration - 1, "=" * 32)
        print("Allowed Tools", [t.name for t in data.request.allowed_tools])
        print("LLM Tool Choice", data.request.tool_choice)
        print("Can Stop?", data.request.can_stop)
        print("")

    result = await agent.run(
        prompt="Tell me about the history of Prague and provide the current weather conditions."
    ).on("start", on_start_request)
    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())

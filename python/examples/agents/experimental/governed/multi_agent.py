import asyncio
import sys
import traceback

from beeai_framework.agents.governed.agent import GovernedAgent
from beeai_framework.agents.governed.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.agents.governed.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.logger.middleware import GlobalLoggerMiddleware
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool
from examples.helpers.io import ConsoleReader

reader = ConsoleReader()


async def main() -> None:
    researcher = GovernedAgent(
        name="ResearchAgent",
        description="Expert for detailed research on topics",
        model=ChatModel.from_name("ollama:granite3.3:8b"),
        memory=UnconstrainedMemory(),
        tools=[ThinkTool(), WikipediaTool(), DuckDuckGoSearchTool()],
        requirements=[
            AskPermissionRequirement(exclude=ThinkTool),
            ConditionalRequirement("Wikipedia", min_invocations=1),
            ConditionalRequirement("DuckDuckGo", only_after="Wikipedia", max_invocations=2),
        ],
        role="Research Specialist",
        instructions="You are an expert researcher. Always verify facts through reliable sources and provide comprehensive, accurate information with citations.",  # noqa: E501
    )

    meteorologist = GovernedAgent(
        name="WeatherAgent",
        description="Weather forecast specialist",
        model=ChatModel.from_name("ollama:granite3.3:8b"),
        memory=UnconstrainedMemory(),
        tools=[ThinkTool(), OpenMeteoTool()],
        requirements=[
            ConditionalRequirement(ThinkTool, force_at_step=0, can_be_used_in_row=False),
            AskPermissionRequirement("OpenMeteoTool", remember_choices=True, hide_disallowed=False),
            ConditionalRequirement("OpenMeteoTool", force_at_step=2, min_invocations=1),
        ],
        role="Weather Forecaster",
        instructions="You are a meteorological expert. Provide accurate weather forecasts with specific temperature, precipitation, and wind details, along with relevant context about the weather patterns.",  # noqa: E501
    )

    manager = GovernedAgent(
        model=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[
            ThinkTool(),
            HandoffTool(
                researcher,
                name="ResearcherDepartment",
                description="Transfer to Research Department, they are expert in doing research on a given topic.",
            ),
            HandoffTool(
                meteorologist,
                name="MeteorologistDepartment",
                description="Transfer to Meteorologist Department, they are expert in providing weather forecasts.",
            ),
        ],
        requirements=[
            ConditionalRequirement(ThinkTool),
            AskPermissionRequirement(["ResearcherDepartment", "MeteorologistDepartment"], remember_choices=False),
        ],
        instructions=(
            "You are a project manager who coordinates complex tasks by delegating to specialized experts. "
            "For research queries, delegate to the ResearchAgent. "
            "For weather inquiries, delegate to the WeatherAgent. "
            "If the request is unclear, ask for clarification before delegating. "
            "Synthesize information from different sources to provide comprehensive answers."
        ),
    )

    prompt = "Can you tell me about the history of Boston and what the weather is like there today?"
    reader.write("👨‍💻 User :", prompt)

    try:
        response = await manager.run(prompt).middleware(GlobalLoggerMiddleware())
        reader.write("🤖 Agent :", response.result.text)

    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import sys
import traceback

from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    workflow = AgentWorkflow(name="Smart assistant")
    workflow.add_agent(
        name="Researcher",
        instructions="You are a researcher assistant. Respond only if you can provide a useful answer.",
        tools=[WikipediaTool()],
        llm=llm,
    )
    workflow.add_agent(
        name="WeatherForecaster",
        instructions="You are a weather assistant. Respond only if you can provide a useful answer.",
        tools=[OpenMeteoTool()],
        llm=llm,
        execution=AgentExecutionConfig(max_iterations=3, total_max_retries=10, max_retries_per_step=3),
    )
    workflow.add_agent(
        name="Solver",
        instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
        llm=llm,
    )

    prompt = "What is the capital of France and what is the current weather there?"
    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content=prompt))
    response = await workflow.run(messages=memory.messages)
    print(f"result: {response.state.final_answer}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

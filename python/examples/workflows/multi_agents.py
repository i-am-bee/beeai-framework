import asyncio
import traceback

from pydantic import ValidationError

from beeai_framework.agents.bee.agent import BeeAgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow
from beeai_framework.workflows.workflow import WorkflowError


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    try:
        workflow = AgentWorkflow(name="Smart assistant")
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="WeatherForecaster",
                instructions="You are a weather assistant.",
                tools=[OpenMeteoTool()],
                llm=llm,
                execution=BeeAgentExecutionConfig(max_iterations=3, total_max_retries=10, max_retries_per_step=3),
            )
        )
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="Researcher",
                instructions="You are a researcher assistant.",
                tools=[DuckDuckGoSearchTool()],
                llm=llm,
            )
        )
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="Solver",
                instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
                llm=llm,
            )
        )

        prompt = "What is the weather in New York?"
        memory = UnconstrainedMemory()
        await memory.add(UserMessage(content=prompt))
        response = await workflow.run(messages=memory.messages)
        print(f"result: {response.state.final_answer}")

    except WorkflowError:
        traceback.print_exc()
    except ValidationError:
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

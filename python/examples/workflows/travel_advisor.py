import asyncio
import sys
import traceback

from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput


async def main() -> None:
    destination = input("Enter your travel destination (e.g., Boston, MA): ")
    travel_dates = input("Enter your travel dates (e.g., Mar 23-25, 2025): ")

    llm = ChatModel.from_name("ollama:llama3.1")

    workflow = AgentWorkflow(name="Travel Advisor")

    workflow.add_agent(
        name="Weather Forecaster",
        role="A diligent weather forecaster",
        instructions="You specialize in reporting on the weather.",
        tools=[OpenMeteoTool()],
        llm=llm,
    )

    workflow.add_agent(
        name="Activity Planner",
        role="An expert in local attractions",
        instructions="You know about interesting activities and would like to share.",
        tools=[DuckDuckGoSearchTool()],
        llm=llm,
    )

    workflow.add_agent(
        name="Travel Advisor",
        role="A travel advisor",
        instructions="""You can synthesize travel details such as weather and recommended activities and provide a coherent summary.""",  # noqa: E501
        llm=llm,
    )

    response = await workflow.run(
        inputs=[
            AgentWorkflowInput(
                prompt=f"Provide a comprehensive weather summary for {destination} from {travel_dates}.",
                expected_output="Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",  # noqa: E501
            ),
            AgentWorkflowInput(
                prompt=f"Search for a set of activities close to {destination} from {travel_dates} that are appropriate in light of the weather conditions.",  # noqa: E501
                expected_output="A list of activities including location and description that are weather appropriate.",
            ),
            AgentWorkflowInput(
                prompt=f"Consider the weather report and recommended activities for the trip to {destination} from {travel_dates} and provide a coherent summary.",  # noqa: E501
                expected_output="A summary of the trip that the traveler could take with them. Break it down by day including weather, location and helpful tips.",  # noqa: E501
            ),
        ]
    ).on(
        "success",
        lambda data, event: print(
            f"-> Step '{data.step}' has been completed with the following outcome.\n\n{data.state.final_answer}"
        ),
    )

    print(":earth_africa: Travel Recommendations :earth_africa:")
    print(response.state.final_answer)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

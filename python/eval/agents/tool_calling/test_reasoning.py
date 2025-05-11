import datetime

import pytest
from dateutil.utils import today as get_today
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool, DuckDuckGoSearchToolInput
from beeai_framework.tools.weather import OpenMeteoTool, OpenMeteoToolInput
from eval.agents.tool_calling.utils import invoke_agent, tool_to_tool_call


@pytest.fixture(scope="function")
def agent() -> ToolCallingAgent:
    return ToolCallingAgent(
        llm=ChatModel.from_name("ollama:granite3.2"),
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        abilities=[],
        save_intermediate_steps=True,
    )


def get_next_weekend() -> tuple[datetime.date, datetime.date]:
    today = get_today()
    days_until_saturday = (5 - today.weekday()) % 7 or 7
    saturday = today + datetime.timedelta(days=days_until_saturday)
    sunday = saturday + datetime.timedelta(days=1)
    return saturday.date(), sunday.date()


dataset = EvaluationDataset(
    test_cases=[
        LLMTestCase(
            input="I would like to spend the next weekend in Prague. Can you tell me something about the city and the weather forecast?",  # noqa: E501
            actual_output="",
            expected_tools=[
                tool_to_tool_call(
                    DuckDuckGoSearchTool(),
                    input=DuckDuckGoSearchToolInput(query="prague czech republic"),
                    reasoning="I will use DuckDuckGo to find actual information about Prague",
                ),
                tool_to_tool_call(
                    OpenMeteoTool(),
                    input=OpenMeteoToolInput(
                        location_name="Prague",
                        country="Czech Republic",
                        start_date=get_next_weekend()[0],
                        end_date=get_next_weekend()[1],
                    ),
                    reasoning="I will use OpenMeteo to find weather forecast",
                ),
            ],
            expected_output="""Prague, the capital of the Czech Republic, is a city renowned for its rich history, stunning architecture, and vibrant cultural scene. Often referred to as the "City of a Hundred Spires," Prague's historical center is a UNESCO World Heritage Site, featuring landmarks such as the Gothic-style St. Vitus Cathedral, the medieval Charles Bridge adorned with statues, and the Old Town Square, home to the Astronomical Clock—one of the oldest working astronomical clocks in the world. ​
The city's diverse architectural styles, including Romanesque, Gothic, Renaissance, and Baroque, reflect its evolution over centuries. Notably, Prague served as the capital of the Kingdom of Bohemia and was the residence of several Holy Roman Emperors, most famously Charles IV and Rudolf II.

For your upcoming visit on April 12th and 13th, 2025, the weather in Prague is expected to be pleasant:

Sat, Apr 12, 59°, 39°
Morning low clouds followed by clouds giving way to some sun

Sun, Apr 13, 62°, 40°
Plenty of sunshine
""",  # noqa: E501
        )
    ]
)


@pytest.mark.parametrize("test_case", dataset)
@pytest.mark.asyncio
async def test_reasoning(agent: ToolCallingAgent, test_case: LLMTestCase) -> None:
    await invoke_agent(agent, test_case)
    # hallucination_metric = HallucinationMetric(threshold=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])

import logging

import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool
from eval.agents.tool_calling.utils import invoke_agent


@pytest.fixture(scope="function")
def agent() -> ToolCallingAgent:
    return ToolCallingAgent(
        llm=ChatModel.from_name("ollama:granite3.2"),
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        abilities=[],
        save_intermediate_steps=True,
    )


dataset = EvaluationDataset(
    test_cases=[
        LLMTestCase(input=input, actual_output="", expected_output="Hello. How can I help you?", expected_tools=[])
        for input in ["Hello agent!", "Hello world!", "asdads", "trertrte", "", "Hello", "what can you do?"]
    ]
)


@pytest.mark.parametrize("test_case", dataset)
@pytest.mark.asyncio
async def test_greeting(agent: ToolCallingAgent, test_case: LLMTestCase) -> None:
    logging.disable(logging.FATAL)
    await invoke_agent(agent, test_case)
    print(test_case.actual_output)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.8)
    assert_test(test_case, [answer_relevancy_metric], run_async=True)

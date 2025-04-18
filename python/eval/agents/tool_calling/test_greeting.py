import os

import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from eval.agents.tool_calling.dataset.utils import create_dataset_sync

load_dotenv()


def create_agent() -> ToolCallingAgent:
    return ToolCallingAgent(
        llm=ChatModel.from_name("ollama:granite3.2"),
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        abilities=["reasoning"],
        save_intermediate_steps=True,
    )


def greeting_dataset() -> EvaluationDataset:
    return create_dataset_sync(
        name="greeting",
        agent_factory=create_agent,
        cache=str(os.getenv("EVAL_CACHE_DATASET")).lower() == "true",
        goldens=[
            Golden(
                input=input,
                expected_output=output,
                expected_tools=[],
                comments="The output should be greeting in a given language.",
            )
            for input, output in [
                ("Hello agent!", "Hello. How can I help you?"),
                ("Hello", "Hello. How can I help you?"),
                ("Hola", "Hola. ¿En qué puedo ayudarte?"),
                ("Bonjour", "Bonjour. Comment puis-je vous aider?"),
                ("Hallo", "Hallo. Wie kann ich Ihnen helfen?"),
            ]
        ],
    )


@pytest.mark.parametrize("test_case", greeting_dataset())
async def test_greeting(test_case: LLMTestCase) -> None:
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, include_reason=True)
    assert_test(test_case, [answer_relevancy_metric])

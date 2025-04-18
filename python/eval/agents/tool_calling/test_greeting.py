import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from eval.agents.tool_calling.dataset.utils import create_dataset_sync
from eval.agents.tool_calling.model import DeepEvalLLM

load_dotenv()


def create_agent() -> ToolCallingAgent:
    return ToolCallingAgent(
        llm=ChatModel.from_name("ollama:granite3.2"),
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        # abilities=["reasoning"],
        save_intermediate_steps=True,
    )


def greeting_dataset() -> EvaluationDataset:
    return create_dataset_sync(
        name="greeting",
        agent_factory=create_agent,
        cache=False,
        goldens=[
            Golden(
                input=input,
                expected_output=output,
                expected_tools=[],
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
@pytest.mark.asyncio
async def test_greeting(test_case: LLMTestCase) -> None:
    correctness_metric = GEval(
        name="Correctness",
        criteria="""The assistant's response should include a greeting and an offer to help.
The response must be in the same language as the input. No tools should be used, except for reasoning.""",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.TOOLS_CALLED,
        ],
        model=DeepEvalLLM.from_name("ollama:llama3.1"),
    )

    assert_test(test_case, [correctness_metric])

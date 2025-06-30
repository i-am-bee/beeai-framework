# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCaseParams

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from eval._utils import create_dataset, evaluate_dataset
from eval.agents.requirement._utils import run_agent
from eval.model import DeepEvalLLM


def create_agent() -> RequirementAgent:
    return RequirementAgent(
        llm=ChatModel.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]),
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        notes=[
            "If the user wants to chitchat, always respond politely."
            "You can communicate in the following languages: English, German, Spanish, French, Czech.",
            "IMPORTANT: The final answer must be in the same language as was the task.",
        ],
    )


async def greeting_dataset() -> EvaluationDataset:
    return await create_dataset(
        name="greeting",
        agent_factory=create_agent,
        agent_run=run_agent,
        goldens=[
            Golden(
                input=input,
                expected_output=output,
                expected_tools=[],
            )
            for input, output in [
                ("Hello!", "Hello! How can I assist you?"),
                ("Ahoj!", "Ahoj! Jak Vám můžu pomoci?"),
                ("Hola", "Hola! ¿En qué puedo ayudarte?"),
                ("Bonjour", "Bonjour! Comment puis-je vous aider?"),
                ("Hallo", "Hallo! Wie kann ich Ihnen helfen?"),
            ]
        ],
    )


@pytest.mark.asyncio
async def test_greeting() -> None:
    dataset = await greeting_dataset()

    correctness_metric = GEval(
        name="Correctness",
        criteria="\n - ".join(
            [
                "The output matches is in the same language as the input.",
                "No tools must be called.",
                "The output must convey the same message as the expected output. (ignore differences in tone, phrasing, stylistic, exclamation marks, additional text or new lines)",  # noqa: E501
            ]
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.TOOLS_CALLED,
            LLMTestCaseParams.EXPECTED_TOOLS,
        ],
        verbose_mode=True,
        model=DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]),
        threshold=0.65,
    )
    metrics: list[BaseMetric] = [correctness_metric]
    evaluate_dataset(dataset, metrics)

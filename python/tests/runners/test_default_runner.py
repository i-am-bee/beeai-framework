# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.agents.react.runners.base import ReActAgentRunnerToolInput
from beeai_framework.agents.react.runners.default.runner import DefaultRunner
from beeai_framework.agents.react.types import (
    ReActAgentInput,
    ReActAgentIterationMeta,
    ReActAgentIterationResult,
    ReActAgentRunInput,
    ReActAgentRunOptions,
)
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import Emitter
from beeai_framework.memory import TokenMemory
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.utils import AbortSignal

"""
E2E Tests
"""


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_runner_init() -> None:
    llm: ChatModel = ChatModel.from_name("ollama:granite3.3:8b")

    input = ReActAgentInput(
        llm=llm,
        tools=[OpenMeteoTool()],
        memory=TokenMemory(llm),
        execution=AgentExecutionConfig(max_iterations=10, max_retries_per_step=3, total_max_retries=10),
    )
    # TODO Figure out run
    runner = DefaultRunner(input=input, options=ReActAgentRunOptions(execution=input.execution, signal=None), run=None)  # type: ignore

    await runner.init(ReActAgentRunInput(prompt="What is the current weather in White Plains?"))

    await runner.tool(
        input=ReActAgentRunnerToolInput(
            state=ReActAgentIterationResult(tool_name="OpenMeteoTool", tool_input={"location_name": "White Plains"}),
            emitter=Emitter(),
            meta=ReActAgentIterationMeta(iteration=0),
            signal=AbortSignal(),
        )
    )

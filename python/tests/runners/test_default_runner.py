# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest

from beeai_framework.agents.react.runners.base import ReActAgentRunnerToolInput
from beeai_framework.agents.react.runners.default.runner import DefaultRunner
from beeai_framework.agents.react.types import (
    ReActAgentInput,
    ReActAgentIterationMeta,
    ReActAgentIterationResult,
    ReActAgentRunInput,
    ReActAgentRunOptions,
)
from beeai_framework.agents.types import (
    AgentExecutionConfig,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.cancellation import AbortSignal
from beeai_framework.emitter import Emitter
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

"""
E2E Tests
"""


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_runner_init() -> None:
    llm: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

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

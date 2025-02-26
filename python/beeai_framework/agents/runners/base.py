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


from abc import ABC, abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass

from beeai_framework.agents.types import (
    BeeAgentRunIteration,
    BeeAgentTemplates,
    BeeInput,
    BeeIterationResult,
    BeeMeta,
    BeeRunInput,
    BeeRunOptions,
)
from beeai_framework.cancellation import AbortSignal
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.emitter.types import EmitterInput
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.tools import ToolOutput
from beeai_framework.utils.counter import RetryCounter


@dataclass
class BeeRunnerLLMInput:
    meta: BeeMeta
    signal: AbortSignal
    emitter: Emitter


@dataclass
class RunnerIteration:
    emitter: Emitter
    state: BeeIterationResult
    meta: BeeMeta
    signal: AbortSignal


@dataclass
class BeeRunnerToolResult:
    output: ToolOutput
    success: bool


@dataclass
class BeeRunnerToolInput:
    state: BeeIterationResult  # TODO BeeIterationToolResult
    meta: BeeMeta
    signal: AbortSignal
    emitter: Emitter


class BaseRunner(ABC):
    def __init__(self, input: BeeInput, options: BeeRunOptions, run: RunContext) -> None:
        self._input = input
        self._options = options
        self._failed_attempts_counter = RetryCounter(
            max_retries=(
                options.execution.max_iterations if options.execution and options.execution.max_iterations else 0
            ),
            error_type=Exception,  # TODO Specific error type
        )

        self._memory: BaseMemory | None = None

        self._iterations: list[BeeAgentRunIteration] = []
        self._failedAttemptsCounter: RetryCounter = RetryCounter(
            error_type=Exception,  # TODO AgentError
            max_retries=(
                options.execution.total_max_retries if options.execution and options.execution.total_max_retries else 0
            ),
        )
        self._run = run

    @property
    def iterations(self) -> list[BeeAgentRunIteration]:
        return self._iterations

    @property
    def memory(self) -> BaseMemory:
        if self._memory is not None:
            return self._memory
        raise Exception("Memory has not been initialized.")

    async def create_iteration(self) -> RunnerIteration:
        meta: BeeMeta = BeeMeta(iteration=len(self._iterations) + 1)
        max_iterations = (
            self._options.execution.max_iterations
            if self._options.execution and self._options.execution.max_iterations
            else 0
        )

        if meta.iteration > max_iterations:
            # TODO: Raise Agent Error with metadata
            # https://github.com/i-am-bee/beeai-framework/blob/aa4d5e6091ed3bab8096492707ceb03d3b03863b/src/agents/bee/runners/base.ts#L70
            raise Exception(f"Agent was not able to resolve the task in {max_iterations} iterations.")

        emitter = self._run.emitter.child(emitter_input=EmitterInput(group_id=f"`iteration-${meta.iteration}"))
        iteration: BeeAgentRunIteration = await self.llm(
            BeeRunnerLLMInput(emitter=emitter, signal=self._run.signal, meta=meta)
        )
        self._iterations.append(iteration)
        return RunnerIteration(emitter=emitter, state=iteration.state, meta=meta, signal=self._run.signal)

    async def init(self, input: BeeRunInput) -> None:
        self._memory = await self.init_memory(input)

    @abstractmethod
    async def llm(self, input: BeeRunnerLLMInput) -> Awaitable[BeeAgentRunIteration]:
        pass

    @abstractmethod
    async def tool(self, input: BeeRunnerToolInput) -> BeeRunnerToolResult:
        pass

    @abstractmethod
    def default_templates(self) -> BeeAgentTemplates:
        pass

    @abstractmethod
    async def init_memory(self, input: BeeRunInput) -> BaseMemory:
        pass

    @property
    def templates(self) -> BeeAgentTemplates:
        # TODO: overrides
        return self.default_templates()

    # TODO: Serialization

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

import math
from abc import ABC, abstractmethod
from typing import Any, Self

from pydantic import BaseModel, InstanceOf

from beeai_framework.agents import AgentError
from beeai_framework.agents.react.events import react_agent_event_types
from beeai_framework.agents.react.types import (
    ReActAgentInput,
    ReActAgentIterationMeta,
    ReActAgentIterationResult,
    ReActAgentRunInput,
    ReActAgentRunIteration,
    ReActAgentRunOptions,
    ReActAgentTemplateFactory,
    ReActAgentTemplates,
)
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import ToolOutput
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.counter import RetryCounter


class ReActAgentRunnerLLMInput(BaseModel):
    meta: ReActAgentIterationMeta
    signal: InstanceOf[AbortSignal]
    emitter: InstanceOf[Emitter]


class ReActAgentRunnerIteration(BaseModel):
    emitter: InstanceOf[Emitter]
    state: InstanceOf[ReActAgentIterationResult]
    meta: ReActAgentIterationMeta
    signal: InstanceOf[AbortSignal]


class ReActAgentRunnerToolResult(BaseModel):
    output: InstanceOf[ToolOutput]
    success: bool


class ReActAgentRunnerToolInput(BaseModel):
    state: InstanceOf[ReActAgentIterationResult]
    meta: ReActAgentIterationMeta
    signal: InstanceOf[AbortSignal]
    emitter: InstanceOf[Emitter]


class BaseRunner(ABC):
    def __init__(self, input: ReActAgentInput, options: ReActAgentRunOptions, run: RunContext) -> None:
        self._input = input
        self._options = options
        self._memory: BaseMemory | None = None
        self._iterations: list[ReActAgentRunIteration] = []
        self._failed_attempts_counter: RetryCounter = RetryCounter(
            error_type=AgentError,
            max_retries=(
                max(
                    options.execution.total_max_retries
                    if options.execution and options.execution.total_max_retries
                    else 0,
                    1,  # we need to handle empty results from LiteLLM
                )
            ),
        )
        self._run = run

    @property
    def iterations(self) -> list[ReActAgentRunIteration]:
        return self._iterations

    @property
    def memory(self) -> BaseMemory:
        if self._memory is not None:
            return self._memory
        raise Exception("Memory has not been initialized.")

    async def create_iteration(self) -> ReActAgentRunnerIteration:
        meta: ReActAgentIterationMeta = ReActAgentIterationMeta(iteration=len(self._iterations) + 1)
        max_iterations = (
            self._options.execution.max_iterations
            if self._options.execution and self._options.execution.max_iterations
            else math.inf
        )

        if meta.iteration > max_iterations:
            raise AgentError(f"Agent was not able to resolve the task in {max_iterations} iterations.")

        emitter = self._run.emitter.child(
            group_id=f"iteration-{meta.iteration}",
            events=react_agent_event_types,
        )
        iteration = await self.llm(ReActAgentRunnerLLMInput(emitter=emitter, signal=self._run.signal, meta=meta))
        self._iterations.append(iteration)
        return ReActAgentRunnerIteration(emitter=emitter, state=iteration.state, meta=meta, signal=self._run.signal)

    async def init(self, input: ReActAgentRunInput) -> None:
        self._memory = await self._init_memory(input)

    @abstractmethod
    async def llm(self, input: ReActAgentRunnerLLMInput) -> ReActAgentRunIteration:
        pass

    @abstractmethod
    async def tool(self, input: ReActAgentRunnerToolInput) -> ReActAgentRunnerToolResult:
        pass

    @abstractmethod
    def default_templates(self) -> ReActAgentTemplates:
        pass

    @abstractmethod
    async def _init_memory(self, input: ReActAgentRunInput) -> BaseMemory:
        pass

    @property
    def templates(self) -> ReActAgentTemplates:
        overrides = self._input.templates or {}
        templates = {}

        for key, default_template in self.default_templates().model_dump().items():
            override: PromptTemplate[Any] | ReActAgentTemplateFactory = overrides.get(key) or default_template
            if isinstance(override, PromptTemplate):
                templates[key] = override
                continue
            templates[key] = override(default_template) or default_template
        return ReActAgentTemplates(**templates)

    async def clone(self) -> Self:
        cloned = type(self)(self._input.model_copy(), self._options.model_copy(), await self._run.clone())
        cloned._memory = await self._memory.clone() if self._memory else None
        cloned._failed_attempts_counter = await self._failed_attempts_counter.clone()
        return cloned

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

from abc import abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from functools import cached_property
from typing import Any

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.agents.errors import AgentError
from beeai_framework.agents.tool_calling.utils import ToolCallCheckerConfig
from beeai_framework.agents.types import AgentContext, AgentMeta, AgentRunOutput
from beeai_framework.backend import AnyMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.context import Run, RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.runnable import Runnable
from beeai_framework.tools import AnyTool
from beeai_framework.utils import AbortSignal

Input = TypeVar("Input", bound=str | AnyMessage | list[AnyMessage], default=str)
Output = TypeVar("Output", bound=BaseModel, default=AgentRunOutput)
Context = TypeVar("Context", bound=AgentContext, default=AgentContext)


class BaseAgent(Runnable[Input, Context, Output]):
    """An abstract agent."""

    def __init__(
        self,
        *,
        llm: ChatModel,
        name: str | None = None,
        description: str | None = None,
        role: str | None = None,
        instructions: str | list[str] | None = None,
        notes: str | list[str] | None = None,
        tools: list[AnyTool] | None = None,
        tool_call_checker: ToolCallCheckerConfig | bool = True,
        memory: BaseMemory | None = None,
        middlewares: Sequence[RunMiddlewareType] | None = None,
        stream: bool = True,
    ) -> None:
        super().__init__()
        self._is_running = False
        self._llm = llm
        self._memory = memory or UnconstrainedMemory()
        self._tools = tools or []
        self._tool_call_checker = tool_call_checker
        self._meta = AgentMeta(name=name or self.__class__.__name__, description=description or "", tools=self._tools)
        self._role = role
        self._instructions = instructions
        self._notes = notes
        self.middlewares = middlewares or []
        self._stream = stream

    @abstractmethod
    def _create_emitter(self) -> Emitter:
        pass

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    def destroy(self) -> None:
        self.emitter.destroy()

    @property
    @abstractmethod
    def memory(self) -> BaseMemory:
        pass

    @memory.setter
    @abstractmethod
    def memory(self, memory: BaseMemory) -> None:
        pass

    @property
    def meta(self) -> AgentMeta:
        return self._meta

    def _to_run(
        self,
        fn: Callable[[RunContext], Awaitable[Output]],
        *,
        signal: AbortSignal | None,
        run_params: dict[str, Any] | None = None,
    ) -> Run[Output]:
        if self._is_running:
            raise RuntimeError("Agent is already running!")

        async def handler(context: RunContext) -> Output:
            try:
                self._is_running = True
                return await fn(context)
            except Exception as e:
                raise AgentError.ensure(e)
            finally:
                self._is_running = False

        return RunContext.enter(
            self,
            handler,
            signal=signal,
            run_params=run_params,
        ).middleware(*self.middlewares)


AnyAgent = BaseAgent[Any, Any, Any]

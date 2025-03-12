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
from collections.abc import Awaitable, Callable
from functools import cached_property, wraps
from typing import Any, Generic, ParamSpec, TypeVar

from pydantic import BaseModel

from beeai_framework.agents.errors import AgentError
from beeai_framework.agents.types import AgentMeta
from beeai_framework.cancellation import AbortSignal
from beeai_framework.context import Run, RunContext, RunContextInput, RunInstance
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory

TInput = TypeVar("TInput", bound=BaseModel)
TOptions = TypeVar("TOptions", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)
P = ParamSpec("P")


class BaseAgent(ABC, Generic[TInput, TOptions, TOutput]):
    _run_context: RunContext | None = None

    @abstractmethod
    def _create_emitter(self) -> Emitter:
        pass

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Run[TOutput]:
        pass

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
        return AgentMeta(
            name=self.__class__.__name__,
            description="",
            tools=[],
        )


def run_context(fn: Callable[P, Awaitable[TOutput]]) -> Callable[P, Run[TOutput]]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Run[TOutput]:
        self = args[0]
        assert isinstance(self, BaseAgent)

        if self._run_context:
            raise RuntimeError("Agent is already running!")

        async def handler(context: RunContext) -> TOutput:
            try:
                self._run_context = context
                return await fn(*args, **kwargs)
            except Exception as e:
                raise AgentError.ensure(e)
            finally:
                self._run_context = None

        signal = kwargs.get("signal")
        assert isinstance(signal, AbortSignal | None)

        return RunContext.enter(
            RunInstance(emitter=self.emitter),
            RunContextInput(
                signal=signal,
                params=(*args, kwargs),
            ),
            handler,
        )

    return wrapper


AnyAgent = BaseAgent[Any, Any, Any]

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
from functools import cached_property
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from beeai_framework.agents.errors import AgentError
from beeai_framework.agents.types import AgentMeta
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.utils import AbortSignal

TOutput = TypeVar("TOutput", bound=BaseModel)


class BaseAgent(ABC, Generic[TOutput]):
    def __init__(self) -> None:
        super().__init__()
        self._is_running = False

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

    def _to_run(
        self,
        fn: Callable[[RunContext], Awaitable[TOutput]],
        *,
        signal: AbortSignal | None,
        run_params: dict[str, Any] | None = None,
    ) -> Run[TOutput]:
        if self._is_running:
            raise RuntimeError("Agent is already running!")

        async def handler(context: RunContext) -> TOutput:
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
        )


AnyAgent = BaseAgent[Any]

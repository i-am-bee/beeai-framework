# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from functools import cached_property
from typing import Any

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.agents.errors import AgentError
from beeai_framework.agents.types import AgentExecutionConfig, AgentMeta, AgentRunOutput
from beeai_framework.backend.message import UserMessage
from beeai_framework.context import Run, RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.runnable import Runnable
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.models import ModelLike

Input = TypeVar("Input", bound=str | UserMessage | ModelLike, default=str)
Output = TypeVar("Output", bound=BaseModel, default=AgentRunOutput)
Config = TypeVar("Config", bound=AgentExecutionConfig, default=AgentExecutionConfig)


class BaseAgent(Runnable[Input, Output, Config]):
    def __init__(self) -> None:
        super().__init__()
        self._is_running = False
        self.middlewares: list[RunMiddlewareType] = []

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
        return AgentMeta(
            name=self.__class__.__name__,
            description="",
            tools=[],
        )

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

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from functools import cached_property
from typing import Any, Unpack

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.agents.errors import AgentError
from beeai_framework.agents.types import AgentMeta
from beeai_framework.backend import AnyMessage
from beeai_framework.context import Run, RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput
from beeai_framework.utils import AbortSignal


class AgentOptions(RunnableOptions, total=False):
    """Agent options."""

    expected_output: str | type[BaseModel]
    """
    Instruction for steering the agent towards an expected output format.
    This can be a Pydantic model for structured output decoding and validation.
    """

    total_max_retries: int
    """
    Maximum number of model retries.
    """

    max_retries_per_step: int
    """
    Maximum number of model retries per step.
    """

    max_iterations: int
    """
    Maximum number of iterations.
    """

    backstory: str
    """
    Additional piece of information or background for the agent.
    """


class AgentOutput(RunnableOutput):
    """Agent output."""

    output_structured: Any | BaseModel = None
    """The formatted output returned by the agent."""


R = TypeVar("R", default=AgentOutput, bound=AgentOutput)


class BaseAgent(Runnable[R]):
    """An abstract agent."""

    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self._is_running = False

    @abstractmethod
    def run(self, input: str | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]) -> Run[R]:
        """Execute an agent, specializing the runnable interface to accept a string as input
        in addition to a list of messages.

        Args:
            input: The input to the agent
            signal: The agent abort signal
            context: A dictionary that can be used to pass additional context to the agent

        Returns:
            The agent output.
        """
        pass

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
        fn: Callable[[RunContext], Awaitable[R]],
        *,
        signal: AbortSignal | None,
        run_params: dict[str, Any] | None = None,
    ) -> Run[R]:
        if self._is_running:
            raise RuntimeError("Agent is already running!")

        async def handler(context: RunContext) -> R:
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


AnyAgent = BaseAgent[Any]

# T = TypeVar("T")
# P = ParamSpec("P")

# def text_input(func: Callable[P, T]) -> Callable[P, T]:
#     """
#     Decorator that converts list[AnyMessage] input to str by taking the last element.
#     If input is already str, passes it through unchanged.
#     """

#     @wraps(func)
#     def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
#         """Convert list[AnyMessage] to str by taking the last element."""
#         self = args[0] if args else None
#         if not isinstance(self, BaseAgent):
#             raise TypeError("The first argument of an Agent's run method must be an Agent instance.")
#         input: Any = args[1] if args and len(args) > 1 else None
#         text_input = input if isinstance(input, str) else (input[-1].text if input else "")
#         new_args = (args[0], text_input) + args[2:]
#         return func(*new_args, **kwargs)  # type: ignore[arg-type]  # We preserve the original arg structure

#     return wrapper

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import functools
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict, Unpack

from pydantic import BaseModel, ConfigDict
from typing_extensions import ParamSpec, TypeVar

from beeai_framework.backend import AnyMessage
from beeai_framework.context import Run, RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.dicts import exclude_keys

T = TypeVar("T")
P = ParamSpec("P")


class RunnableOptions(TypedDict, total=False):
    """Options for a runnable."""

    signal: AbortSignal
    """The runnable's abort signal data"""

    context: dict[str, Any]
    """Context can be used to pass additional context to runnable"""


class RunnableOutput(BaseModel):
    """Runnable output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: list[AnyMessage]
    """The runnable output"""

    context: dict[str, Any] | None = None
    """Context can be used to return additional data by runnable"""


class Runnable(ABC):
    """A unit of work that can be invoked using a stable interface.

    Attributes:
        _middlewares: The list of middleware to be used when executing the runnable.
    """

    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__()
        self._middlewares = middlewares or []

    @abstractmethod
    def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> Run[RunnableOutput]:
        """ "Execute the runnable.

        Args:
            input: The input to the runnable
            signal: The runnable abort signal
            context: A dictionary that can be used to pass additional context to the runnable

        Returns:
            The runnable output.
        """
        pass

    @property
    @abstractmethod
    def emitter(self) -> Emitter:
        """The event emitter for the runnable."""
        pass

    @property
    def middlewares(self) -> list[RunMiddlewareType]:
        """The list of middleware to be used when executing the runnable."""
        return self._middlewares


def runnable_entry(handler: Callable[P, Awaitable[T]]) -> Callable[P, Run[T]]:
    """A decorator that wraps the runnable into an execution context.

    For example:

        @runnable_entry
        async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
            ctx = RunContext.get()
            # ... runnable logic ...
            return RunnableOutput(output=...)
    """

    @functools.wraps(handler)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Run[T]:
        """Wrapper that automates the call to RunContext.enter()."""

        async def inner(_: RunContext) -> T:
            return await handler(*args, **kwargs)

        self = args[0] if args else None
        if not isinstance(self, Runnable):
            raise TypeError("The first argument of a runnable must be a Runnable instance.")

        runnable_kwargs: RunnableOptions = kwargs  # type: ignore
        return (
            RunContext.enter(
                self,
                inner,
                signal=runnable_kwargs.get("signal", None),
                run_params={"input": args[1], **exclude_keys(kwargs, {"signal", "input"})},
            )
            .middleware(*self.middlewares)
            .context(runnable_kwargs.get("context") or {})
        )

    return wrapper

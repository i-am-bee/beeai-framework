# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from functools import cached_property
from typing import Any, Generic, Self

from pydantic import BaseModel, Field
from typing_extensions import TypeVar

from beeai_framework.agents.experimental.requirements._utils import (
    MultiTargetType,
    _assert_targets_exist,
    _extract_targets,
)
from beeai_framework.context import Run, RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import AnyTool
from beeai_framework.utils import MaybeAsync
from beeai_framework.utils.strings import to_safe_word


class Rule(BaseModel):
    target: str = Field(..., description="A tool that the requirement apply to.")
    allowed: bool = Field(True, description="Can the agent use the tool?")
    prevent_stop: bool = Field(False, description="Prevent the agent from terminating.")
    forced: bool = Field(False, description="Must the agent use the tool?")
    hidden: bool = Field(False, description="Completely omit the tool.")

    def __bool__(self) -> bool:
        return self.allowed


T = TypeVar("T", bound=Any, default=Any)


class Requirement(ABC, Generic[T]):
    name: str
    state: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._priority = 10
        self.enabled = True
        self.state = {}
        self.middlewares: list[RunMiddlewareType] = []

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["requirement", to_safe_word(self.name)],
            creator=self,
            events={},
        )

    @property
    def priority(self) -> int:
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Priority must be a positive integer.")

        self._priority = value

    @abstractmethod
    def run(self, state: T) -> Run[list[Rule]]: ...

    def init(self, *, tools: list[AnyTool], ctx: RunContext) -> None:
        pass

    async def clone(self) -> Self:
        instance = type(self).__new__(self.__class__)
        instance.name = self.name
        instance.priority = self.priority
        instance.enabled = self.enabled
        instance.state = self.state.copy()
        return instance


TSelf = TypeVar("TSelf", bound=Requirement[Any])


def run_with_context(
    fn: Callable[
        [TSelf, T, "RunContext"],
        Awaitable[list[Rule]],
    ],
) -> Callable[[TSelf, T], Run[list[Rule]]]:
    def decorated(self: TSelf, input: T) -> Run[list[Rule]]:
        async def handler(context: RunContext) -> list[Rule]:
            return await fn(self, input, context)

        return RunContext.enter(
            self,
            handler,
            run_params=input.model_dump() if isinstance(input, BaseModel) else input,
        ).middleware(*self.middlewares)

    return decorated


RequirementFn = MaybeAsync[[T, RunContext], list[Rule]]


def requirement(
    *,
    name: str | None = None,
    targets: MultiTargetType | None = None,
) -> Callable[[RequirementFn], Requirement[T]]:
    def create_requirement(
        fn: RequirementFn,
    ) -> Requirement[Any]:
        req_name = name or fn.__name__
        req_targets = _extract_targets(targets)

        class FunctionRequirement(Requirement[Any]):
            name = req_name or fn.__name__

            @run_with_context
            async def run(self, state: T, context: RunContext) -> list[Rule]:
                result = fn(state, context)
                if inspect.isawaitable(result):
                    return await result
                else:
                    return result

            def init(self, *, tools: list[AnyTool], ctx: RunContext) -> None:
                _assert_targets_exist(targets=tools, allowed=req_targets)

        return FunctionRequirement()

    return create_requirement


class RequirementError(FrameworkError):
    def __init__(
        self,
        message: str = "Framework error",
        *,
        requirement: Requirement[Any] | None = None,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause, context=context)
        self._requirement = requirement

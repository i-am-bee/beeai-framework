# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from beeai_framework.runnable import RunnableOutput
from pydantic import BaseModel

ControllerFunction = Callable[..., Awaitable[Any]]
BooleanControllerFunction = Callable[..., Awaitable[bool]]
AsyncStepFunction = Callable[..., Coroutine[Any, Any, Any]]


class StepCondition(BaseModel):
    fn: ControllerFunction
    key: Any


class StepLoopCondition(BaseModel):
    fn: BooleanControllerFunction


# # Start of workflow
# StartStepMethod = Callable[[TWorkflow, list[AnyMessage], Unpack[RunnableOptions]], Coroutine[Any, Any, Any]]

# # End of workflow
EndStepMethod = Callable[..., Coroutine[Any, Any, RunnableOutput]]

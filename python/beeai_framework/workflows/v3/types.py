# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from pydantic import BaseModel

from beeai_framework.runnable import RunnableOutput

ControllerFunction = Callable[..., Awaitable[Any]]
BooleanControllerFunction = Callable[..., Awaitable[bool]]
AsyncStepFunction = Callable[..., Coroutine[Any, Any, Any]]


class BranchCondition(BaseModel):
    fn: ControllerFunction
    key: Any


class StepLoopCondition(BaseModel):
    fn: BooleanControllerFunction


# # End of workflow
EndStepMethod = Callable[..., Coroutine[Any, Any, RunnableOutput]]

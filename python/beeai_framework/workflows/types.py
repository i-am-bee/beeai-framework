# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import field
from typing import Any, Generic, Literal

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.utils import AbortSignal
from beeai_framework.utils.types import MaybeAsync

T = TypeVar("T", bound=BaseModel)
K = TypeVar("K", default=str)


WorkflowReservedStepName = Literal["__start__", "__self__", "__prev__", "__next__", "__end__"]
WorkflowHandler = MaybeAsync[[T], K | WorkflowReservedStepName | None]


class WorkflowRunOptions(BaseModel, Generic[K]):
    start: K | None = None
    signal: AbortSignal | None = None


class WorkflowState(BaseModel, Generic[K]):
    current: K
    prev: K | None = None
    next: K | None = None


class WorkflowStepRes(BaseModel, Generic[T, K]):
    name: K
    state: T


class WorkflowStepDefinition(BaseModel, Generic[T, K]):
    handler: WorkflowHandler[T, K]


class WorkflowRunContext(BaseModel, Generic[T, K]):
    steps: list[WorkflowStepRes[T, K]] = field(default_factory=list)
    signal: AbortSignal
    abort: Callable[[Any], None]


class WorkflowRun(BaseModel, Generic[T, K]):
    state: T
    result: T | None = None
    steps: list[WorkflowStepRes[T, K]] = field(default_factory=list)

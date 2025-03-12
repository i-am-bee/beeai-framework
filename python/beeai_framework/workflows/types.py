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

from collections.abc import Callable
from dataclasses import field
from typing import Any, Generic, Literal

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.cancellation import AbortSignal
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

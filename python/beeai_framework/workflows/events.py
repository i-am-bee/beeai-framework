# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, Literal

from pydantic import BaseModel, InstanceOf
from typing_extensions import TypeVar

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.types import WorkflowRun

T = TypeVar("T", bound=BaseModel)
K = TypeVar("K", default=str)


class WorkflowStartEvent(BaseModel, Generic[T, K]):
    run: WorkflowRun[T, K]
    step: K


class WorkflowSuccessEvent(BaseModel, Generic[T, K]):
    run: WorkflowRun[T, K]
    state: T
    step: K
    next: K | Literal["__end__"]


class WorkflowErrorEvent(BaseModel, Generic[T, K]):
    run: WorkflowRun[T, K]
    step: K | Literal["__end__"]
    error: InstanceOf[FrameworkError]

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from beeai_framework.workflows.v2.decorators.when import Predicate
from beeai_framework.workflows.v2.types import AsyncFunc, DependencyType


class WorkflowStepExecution(BaseModel):
    inputs: tuple[Any | None, ...]
    output: Any | None
    error: Exception | None
    started_at: datetime | None
    ended_at: datetime | None
    duration: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowStep:
    def __init__(
        self,
        func: AsyncFunc,
        start: bool = False,
        end: bool = False,
        fork: bool = False,
        join: bool = False,
        retries: int = 0,
    ) -> None:
        self._func = func
        self._name = func.__name__

        self._is_start = start
        self._is_end = end
        self._is_fork = fork
        self._is_join = join
        self._retries = retries

        self.forked: list[WorkflowStep] = []
        self._dependencies: list[WorkflowStep] = []
        self._dependents: list[WorkflowStep] = []
        self.completed_dependencies: list[WorkflowStep] = []
        self.inputs: list[Any | None] = []

        self.type: DependencyType = "AND"
        self.completed_event = asyncio.Event()
        self._predicates: list[Predicate] = []
        self._executions: list[WorkflowStepExecution] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def func(self) -> AsyncFunc:
        return self._func

    @property
    def is_start(self) -> bool:
        return self._is_start

    @property
    def is_end(self) -> bool:
        return self._is_end

    @property
    def is_fork(self) -> bool:
        return self._is_fork

    @property
    def is_join(self) -> bool:
        return self._is_join

    @property
    def retries(self) -> int:
        return self._retries

    def add_predicate(self, predicate: Predicate) -> None:
        self._predicates.append(predicate)

    @property
    def predicates(self) -> list[Predicate]:
        return self._predicates

    @property
    def executions(self) -> list[WorkflowStepExecution]:
        return self._executions

    def add_dependency(self, dep: "WorkflowStep") -> None:
        self._dependencies.append(dep)
        self.inputs.append(None)
        dep._dependents.append(self)

    @property
    def dependencies(self) -> list["WorkflowStep"]:
        return self._dependencies

    @property
    def dependents(self) -> list["WorkflowStep"]:
        return self._dependents

    def last_execution(self) -> WorkflowStepExecution | None:
        return self.executions[-1] if self.executions else None

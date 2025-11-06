# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from beeai_framework.workflows.v3.types import AsyncStepFunction, ControllerFunction


class WorkflowStep:
    def __init__(
        self,
        func: AsyncStepFunction,
    ) -> None:
        self._func = func
        self._name = func.__name__

        self._upstream: list[WorkflowStep] = []
        self._downstream: list[WorkflowStep | WorkflowBranch | WorkflowLoopUntil] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def func(self) -> AsyncStepFunction:
        return self._func

    def then(self, next_steps: WorkflowStep | list[WorkflowStep]) -> WorkflowStep:
        if isinstance(next_steps, list):
            for step in next_steps:
                self._downstream.append(step)
                step._upstream.append(self)
            return step  # TODO: Returning the last concurrent step is not the answer here
        else:
            self._downstream.append(next_steps)
            next_steps._upstream.append(self)
            return next_steps

    def branch(self, next_steps: dict[Any, WorkflowStep], branch_fn: ControllerFunction) -> None:
        branch = WorkflowBranch(branch_fn=branch_fn, next_steps=next_steps)
        self._downstream.append(branch)
        for step in next_steps.values():
            step._upstream.append(self)

    def loop_until(
        self,
        step: WorkflowStep,
        until_fn: ControllerFunction,
    ) -> WorkflowStep:
        loop = WorkflowLoopUntil(step=step, until_fn=until_fn)
        self._downstream.append(loop)
        step._upstream.append(loop.step)
        return step


class WorkflowLoopUntil(BaseModel):
    until_fn: ControllerFunction
    step: WorkflowStep

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowBranch(BaseModel):
    branch_fn: ControllerFunction
    next_steps: dict[Any, WorkflowStep]

    model_config = ConfigDict(arbitrary_types_allowed=True)

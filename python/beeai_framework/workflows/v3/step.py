# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from beeai_framework.workflows.v3.types import AsyncStepFunction, ControllerFunction


class StepExecutable(ABC):
    """
    The executable element associated with a WorkflowStep.
    Decomposes step mechanics from execution.
    """

    @abstractmethod
    async def execute(self) -> Any:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def requeue(self) -> bool:
        return False


class EmptyStepExecutable(StepExecutable):
    """
    Empty step, does nothing.
    """

    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def execute(self) -> Any:
        pass


class AsyncFuncStepExecutable(StepExecutable):
    """
    Executes an async function/method.
    """

    def __init__(
        self,
        func: AsyncStepFunction | None = None,
    ) -> None:
        super().__init__()
        self._func = func

    @property
    def name(self) -> str:
        if self._func:
            return self._func.__name__
        return ""

    async def execute(self) -> Any:
        if self._func:
            return await self._func()


class ConditionalStepExecutable(StepExecutable):
    """
    Conditionally executes one from a set of steps.
    """

    def __init__(self, steps: dict[Any, WorkflowStep], cond_fn: ControllerFunction) -> None:
        super().__init__()
        self._steps = steps
        self._cond_fn = cond_fn

    async def execute(self) -> Any:
        key = self._cond_fn()
        if key in self._steps:
            return await self._steps[key].execute()

    @property
    def name(self) -> str:
        return "/".join([s.name for s in self._steps.values()])


class LoopUntilStepExecutable(StepExecutable):
    """
    Executes a step in a loop. Looping is handled by the step execution via the requeue.
    """

    def __init__(self, step: WorkflowStep, until_fn: ControllerFunction) -> None:
        self._step = step
        self._until_fn = until_fn

    async def execute(self) -> Any:
        return await self._step.execute()

    @property
    def name(self) -> str:
        return self._step.name

    def requeue(self) -> bool:
        return self._until_fn()


class WorkflowStep:
    def __init__(self, executable: StepExecutable) -> None:
        self._step_executable = executable
        self._upstream: list[WorkflowStep] = []
        self._downstream: list[WorkflowStep] = []

    def execute(self) -> Any:
        return self._step_executable.execute()

    @property
    def name(self) -> Any:
        return self._step_executable.name

    def requeue(self) -> bool:
        return self._step_executable.requeue()

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

    def branch(self, steps: dict[Any, WorkflowStep], branch_fn: ControllerFunction) -> WorkflowStep:
        branch_step = WorkflowStep(executable=ConditionalStepExecutable(steps=steps, cond_fn=branch_fn))
        self._downstream.append(branch_step)
        for step in steps.values():
            step._upstream.append(self)
        return branch_step

    def loop_until(
        self,
        step: WorkflowStep,
        until_fn: ControllerFunction,
    ) -> WorkflowStep:
        loop_step = WorkflowStep(executable=LoopUntilStepExecutable(step=step, until_fn=until_fn))
        self._downstream.append(loop_step)
        loop_step._upstream.append(self)
        return loop_step

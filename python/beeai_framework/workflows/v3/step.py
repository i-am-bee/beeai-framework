# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from beeai_framework.workflows.v3.types import AsyncStepFunction, BooleanControllerFunction, ControllerFunction
from beeai_framework.workflows.v3.util import run_callable


class WorkflowChainable:
    def __init__(self, frontier: list[WorkflowStep] | None = None) -> None:
        self._frontier = frontier or []

    def then(self, next_steps: WorkflowStep | list[WorkflowStep]) -> WorkflowChainable:
        if not isinstance(next_steps, list):
            next_steps = [next_steps]

        for prev in self._frontier:
            for nxt in next_steps:
                prev.downstream.append(nxt)
                nxt.upstream.append(prev)

        return WorkflowChainable(next_steps)

    def branch(self, steps: dict[Any, WorkflowStep], branch_fn: ControllerFunction) -> WorkflowChainable:
        branch_step = ConditionalWorkflowStep(steps=steps, cond_fn=branch_fn)

        for prev in self._frontier:
            prev.downstream.append(branch_step)

        for step in steps.values():
            step.upstream.append(branch_step)

        return WorkflowChainable([branch_step])

    def loop_until(
        self,
        step: WorkflowStep,
        until_fn: BooleanControllerFunction,
    ) -> WorkflowChainable:
        loop_step = LoopUntilWorkflowStep(step=step, until_fn=until_fn)

        for prev in self._frontier:
            prev.downstream.append(loop_step)

        step.upstream.append(loop_step)

        return WorkflowChainable([loop_step])


class WorkflowStep(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._upstream: list[WorkflowStep] = []
        self._downstream: list[WorkflowStep] = []

    @property
    def upstream(self) -> list[WorkflowStep]:
        return self._upstream

    @property
    def downstream(self) -> list[WorkflowStep]:
        return self._downstream

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def execute(self, *inputs: Any) -> Any:
        pass

    @property
    def result(self) -> Any:
        return None

    def requeue(self) -> bool:
        return False


class EmptyWorkflowStep(WorkflowStep):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    async def execute(self) -> Any:
        pass

    @property
    def name(self) -> str:
        return self._name


class FuncWorkflowStep(WorkflowStep):
    """
    Executes an async function/method.
    """

    def __init__(
        self,
        func: AsyncStepFunction | None = None,
    ) -> None:
        super().__init__()
        self._func = func
        self._result: Any = None

    @property
    def name(self) -> str:
        if self._func:
            return self._func.__name__
        return ""

    async def execute(self, *inputs: Any) -> Any:
        if self._func:
            self._result = await run_callable(self._func, *inputs)
            return self._result

    @property
    def result(self) -> Any:
        return self._result


class ConditionalWorkflowStep(WorkflowStep):
    """
    Conditionally executes one from a set of steps.
    """

    def __init__(self, steps: dict[Any, WorkflowStep], cond_fn: ControllerFunction) -> None:
        super().__init__()
        self._steps = steps
        self._cond_fn = cond_fn
        self._last_executed_step: WorkflowStep | None = None

    async def execute(self, *inputs: Any) -> Any:
        key = self._cond_fn(*inputs)
        if key in self._steps:
            self._last_executed_step = self._steps[key]
            return await self._last_executed_step.execute(*inputs)

    @property
    def result(self) -> Any:
        if self._last_executed_step:
            return self._last_executed_step.result
        return None

    @property
    def name(self) -> str:
        return "/".join([s.name for s in self._steps.values()])


class LoopUntilWorkflowStep(WorkflowStep):
    """
    Executes a step in a loop. Looping is handled by the step execution via the requeue.
    """

    def __init__(self, step: WorkflowStep, until_fn: BooleanControllerFunction) -> None:
        super().__init__()
        self._step = step
        self._until_fn = until_fn

    async def execute(self, *inputs: Any) -> Any:
        return await self._step.execute(*inputs)

    @property
    def result(self) -> Any:
        return self._step.result

    @property
    def name(self) -> str:
        return self._step.name

    def requeue(self) -> bool:
        return self._until_fn()

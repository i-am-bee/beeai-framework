# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from beeai_framework.workflows.v3.types import (
    AsyncStepFunction,
    BooleanControllerFunction,
    ControllerFunction,
    StepCondition,
    StepLoopCondition,
)
from beeai_framework.workflows.v3.util import run_callable


class WorkflowBuilder:
    def __init__(self, frontier: list[WorkflowStep] | None = None) -> None:
        self._frontier = frontier or []

    def then(self, next_steps: WorkflowStep | list[WorkflowStep]) -> WorkflowBuilder:
        if not isinstance(next_steps, list):
            next_steps = [next_steps]

        for prev in self._frontier:
            for nxt in next_steps:
                prev.add_downstream(nxt)
                nxt.add_upstream(prev)

        return WorkflowBuilder(next_steps)

    def branch(self, steps: dict[Any, WorkflowStep], branch_fn: ControllerFunction) -> WorkflowBuilder:
        for key, step in steps.items():
            # Apply guard to each step
            step.guard_condition = StepCondition(fn=branch_fn, key=key)

        for prev in self._frontier:
            for nxt in steps.values():
                prev.add_downstream(nxt)
                nxt.add_upstream(prev)

        return WorkflowBuilder(list(steps.values()))

    def loop_until(
        self,
        step: WorkflowStep,
        until_fn: BooleanControllerFunction,
    ) -> WorkflowBuilder:
        step.loop_condition = StepLoopCondition(
            fn=until_fn,
        )

        for prev in self._frontier:
            prev.add_downstream(step)
            step.add_upstream(prev)

        return WorkflowBuilder([step])


class WorkflowStep(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._upstream: list[WorkflowStep] = []
        self._downstream: list[WorkflowStep] = []
        self.remaining_upstream: int = 0
        self.guard_condition: StepCondition | None = None
        self.loop_condition: StepLoopCondition | None = None

    def add_upstream(self, step: WorkflowStep) -> None:
        self._upstream.append(step)
        self.remaining_upstream = len(self._upstream)

    def add_downstream(self, step: WorkflowStep) -> None:
        self._downstream.append(step)

    @property
    def upstream(self) -> list[WorkflowStep]:
        return self._upstream

    @property
    def downstream(self) -> list[WorkflowStep]:
        return self._downstream

    async def condition(self, *inputs: Any) -> bool:
        if self.guard_condition:
            return bool(await run_callable(self.guard_condition.fn, inputs) == self.guard_condition.key)
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def execute(self, *inputs: Any) -> Any:
        pass

    @property
    def result(self) -> Any:
        return None

    async def requeue(self, *inputs: Any) -> bool:
        if self.loop_condition:
            return bool(await run_callable(self.loop_condition.fn, inputs))
        return False


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


class StartWorkflowStep(FuncWorkflowStep):
    pass


class EndWorkflowStep(FuncWorkflowStep):
    pass

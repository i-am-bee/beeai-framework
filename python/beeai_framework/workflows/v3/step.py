# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from beeai_framework.workflows.v3.types import (
    AsyncStepFunction,
    BooleanControllerFunction,
    BranchCondition,
    ControllerFunction,
    StepLoopCondition,
)
from beeai_framework.workflows.v3.util import run_callable


class WorkflowBuilder:
    def __init__(self, frontier: list[WorkflowStep] | None = None) -> None:
        self._frontier = frontier or []

    def then(self, next_steps: WorkflowStep | list[WorkflowStep]) -> WorkflowBuilder:
        if isinstance(next_steps, list):
            for prev in self._frontier:
                join_step = JoinWorkflowStep()

                for nxt in next_steps:
                    prev.add_downstream_step(nxt)
                    nxt.add_upstream_step(prev)

                    nxt.add_downstream_step(join_step)
                    join_step.add_upstream_step(nxt)

            return WorkflowBuilder([join_step])
        else:
            next_steps = [next_steps]
            for prev in self._frontier:
                for nxt in next_steps:
                    prev.add_downstream_step(nxt)
                    nxt.add_upstream_step(prev)

            return WorkflowBuilder(next_steps)

    def branch(self, steps: dict[Any, WorkflowStep], branch_fn: ControllerFunction) -> WorkflowBuilder:
        for prev in self._frontier:
            join_step = JoinWorkflowStep()
            for key, nxt in steps.items():
                prev.add_downstream_step(nxt, BranchCondition(fn=branch_fn, key=key))
                nxt.add_upstream_step(prev)
                nxt.add_downstream_step(join_step, optional=True)
                join_step.add_upstream_step(nxt, optional=True)

        return WorkflowBuilder([join_step])

    def loop_until(
        self,
        step: WorkflowStep,
        until_fn: BooleanControllerFunction,
    ) -> WorkflowBuilder:
        step.loop_condition = StepLoopCondition(
            fn=until_fn,
        )

        for prev in self._frontier:
            prev.add_downstream_step(step)
            step.add_upstream_step(prev)

        return WorkflowBuilder([step])


class WorkflowEdge:
    def __init__(
        self,
        source: WorkflowStep,
        target: WorkflowStep,
        condition: BranchCondition | None = None,
        optional: bool = False,
    ) -> None:
        self.source = source
        self.target = target
        self.condition = condition
        self.optional = optional


class WorkflowStep(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._upstream_edges: list[WorkflowEdge] = []
        self._downstream_edges: list[WorkflowEdge] = []

        self.has_executed: bool = False
        self.loop_condition: StepLoopCondition | None = None

    def add_upstream_step(self, step: WorkflowStep, optional: bool = False) -> None:
        self._upstream_edges.append(WorkflowEdge(step, self, optional=optional))

    def add_downstream_step(
        self, step: WorkflowStep, condition: BranchCondition | None = None, optional: bool = False
    ) -> None:
        self._downstream_edges.append(WorkflowEdge(self, step, condition=condition, optional=optional))

    @property
    def upstream_steps(self) -> list[WorkflowStep]:
        return [edge.source for edge in self._upstream_edges]

    @property
    def downstream_steps(self) -> list[WorkflowStep]:
        return [edge.target for edge in self._downstream_edges]

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


class JoinWorkflowStep(WorkflowStep):
    @property
    def name(self) -> str:
        return "__join__"

    async def execute(self, *inputs: Any) -> Any:
        pass

    @property
    def result(self) -> Any:
        res = []
        for up in self.upstream_steps:
            if up.has_executed:
                res.append(up.result)
        return res


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

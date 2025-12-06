# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from beeai_framework.workflows.v3.types import (
    AsyncStepFunction,
    BranchCondition,
    ControllerFunction,
    StepLoopCondition,
)
from beeai_framework.workflows.v3.util import run_callable


class WorkflowBuilder:
    def __init__(self, root: WorkflowStep) -> None:
        self._root = root
        self._current_step = root
        self._steps: set[WorkflowStep] = {self._current_step}

    def then(self, next_steps: WorkflowStep | list[WorkflowStep]) -> WorkflowBuilder:
        if isinstance(next_steps, list):
            join_step = JoinWorkflowStep()

            for nxt in next_steps:
                self._current_step.add_downstream_step(nxt)
                nxt.add_upstream_step(self._current_step)

                nxt.add_downstream_step(join_step)
                join_step.add_upstream_step(nxt, edge_type="and")

                self._steps.add(nxt)

            self._current_step = join_step
            return self
        else:
            next_steps = [next_steps]
            # for prev in self._frontier:
            for nxt in next_steps:
                self._current_step.add_downstream_step(nxt)
                nxt.add_upstream_step(self._current_step)

                self._steps.add(nxt)

            self._current_step = nxt
            return self

    def branch(
        self, steps: dict[Any, WorkflowStep], branch_fn: ControllerFunction | None = None
    ) -> AfterWorkflowBuilder:
        # for prev in self._frontier:
        # join_step = JoinWorkflowStep()
        for key, nxt in steps.items():
            self._current_step.add_downstream_step(nxt, BranchCondition(fn=branch_fn, key=key))
            nxt.add_upstream_step(self._current_step)
            self._steps.add(nxt)
            # nxt.add_downstream_step(join_step, optional=True)
            # join_step.add_upstream_step(nxt, optional=True)
        return AfterWorkflowBuilder(builder=self)

    def after(self, step: WorkflowStep) -> WorkflowBuilder:
        assert step in self._steps
        self._current_step = step
        return self

    # TODO
    # def loop_until(
    #     self,
    #     step: WorkflowStep,
    #     until_fn: BooleanControllerFunction,
    # ) -> WorkflowBuilder:
    #     step.loop_condition = StepLoopCondition(
    #         fn=until_fn,
    #     )

    #     for prev in self._frontier:
    #         prev.add_downstream_step(step)
    #         step.add_upstream_step(prev)

    #     return WorkflowBuilder([step])


class AfterWorkflowBuilder:
    def __init__(self, builder: WorkflowBuilder) -> None:
        self._builder = builder

    def after(self, step: WorkflowStep) -> WorkflowBuilder:
        return self._builder.after(step)


class WorkflowEdge:
    def __init__(
        self,
        source: WorkflowStep,
        target: WorkflowStep,
        condition: BranchCondition | None = None,
        type: Literal["and", "or"] = "or",
    ) -> None:
        self.source = source
        self.target = target
        self.condition = condition
        self.type: Literal["and", "or"] = type


class WorkflowStep(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._upstream_edges: list[WorkflowEdge] = []
        self._downstream_edges: list[WorkflowEdge] = []
        self.has_executed: bool = False
        self.loop_condition: StepLoopCondition | None = None

    def add_upstream_step(self, step: WorkflowStep, edge_type: Literal["and", "or"] = "or") -> None:
        self._upstream_edges.append(WorkflowEdge(step, self, type=edge_type))

    def add_downstream_step(
        self, step: WorkflowStep, condition: BranchCondition | None = None, edge_type: Literal["and", "or"] = "or"
    ) -> None:
        self._downstream_edges.append(WorkflowEdge(self, step, condition=condition, type=edge_type))

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
            return bool(await run_callable(self.loop_condition.fn, *inputs))
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

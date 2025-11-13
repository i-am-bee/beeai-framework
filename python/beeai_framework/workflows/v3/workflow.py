# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, TypeVar, Unpack, overload

from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import RunContext, RunMiddlewareType
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry
from beeai_framework.workflows.v3.events import (
    workflow_v3_event_types,
)
from beeai_framework.workflows.v3.step import WorkflowBranch, WorkflowLoopUntil, WorkflowStep
from beeai_framework.workflows.v3.types import AsyncStepFunction


def create_step(func: AsyncStepFunction) -> WorkflowStep:
    return WorkflowStep(func=func)


T = TypeVar("T", bound="Workflow")


class step:  # noqa: N801
    """Descriptor that turns an async method into a WorkflowStep."""

    def __init__(self, func: AsyncStepFunction) -> None:
        self.func = func
        self.name = func.__name__

    def __set_name__(self, owner: T, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type[T]) -> "step": ...
    @overload
    def __get__(self, instance: T, owner: type[T]) -> WorkflowStep: ...

    def __get__(self, instance: T | None, owner: type[T]) -> Any:
        if instance is None:
            return self  # accessed on class, not instance

        cache_name = f"__workflow_cache_{self.name}"
        if not hasattr(instance, cache_name):
            bound_func = self.func.__get__(instance, owner)
            setattr(instance, cache_name, WorkflowStep(bound_func))
        return getattr(instance, cache_name)


class Workflow(Runnable[RunnableOutput], ABC):

    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self._messages: list[AnyMessage] = []
        self._context: dict[str, Any] = {}

    @property
    def messages(self) -> list[AnyMessage]:
        return self._messages

    @property
    def context(self) -> dict[str, Any]:
        return self._context

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["workflow", "v3"], creator=self, events=workflow_v3_event_types)

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:

        self._messages = input
        self._context = RunContext.get().context

        # Builds out the execution graph
        # TODO The graph is reconstructed on each run? What would a user expect?
        self._start_step: WorkflowStep = WorkflowStep()
        self.build(self._start_step)

        assert self._start_step is not None

        # Execute the workflow rooted at the start node
        queue: asyncio.Queue[Any] = asyncio.Queue()
        await queue.put(self._start_step)

        # Track running tasks
        tasks: set[asyncio.Task[Any]] = set()
        completed_steps: set[WorkflowStep] = set()

        async def execute_step(step: WorkflowStep | WorkflowBranch | WorkflowLoopUntil) -> None:
            step_to_exec: WorkflowStep | None = None

            # Run the step
            if isinstance(step, WorkflowStep):
                step_to_exec = step
            elif isinstance(step, WorkflowBranch):
                key = step.branch_fn()
                step_to_exec = step.next_steps[key]
            elif isinstance(step, WorkflowLoopUntil):
                step_to_exec = step.step

            assert step_to_exec is not None

            print("Executing:", step_to_exec.name)

            await step_to_exec.execute()

            completed_steps.add(step_to_exec)

            if isinstance(step, WorkflowLoopUntil) and step.until_fn():
                await queue.put(step)
                return

            # Enqueue downstream steps
            for ds in step_to_exec._downstream:
                await queue.put(ds)

        while not queue.empty() or tasks:
            # Drain current queue items into tasks
            while not queue.empty():
                step = await queue.get()
                task = asyncio.create_task(execute_step(step))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                queue.task_done()

            # Wait for at least one task to complete before continuing
            if tasks:
                done, _ = await asyncio.wait(tasks)
                # done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        return RunnableOutput(output=[])

    @abstractmethod
    def build(self, start: WorkflowStep) -> None:
        pass

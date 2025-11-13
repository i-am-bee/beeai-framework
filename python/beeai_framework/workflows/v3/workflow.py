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
from beeai_framework.workflows.v3.step import (
    AsyncFuncStepExecutable,
    EmptyStepExecutable,
    WorkflowStep,
)
from beeai_framework.workflows.v3.types import AsyncStepFunction

T = TypeVar("T", bound="Workflow")


class step:  # noqa: N801
    """
    Descriptor that turns an async method into a WorkflowStep.
    Users can refer to decorated methods and treat as WorkflowStep for composition.
    """

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

        cache_name = f"__workflow_step_cache_{self.name}"
        if not hasattr(instance, cache_name):
            bound_func = self.func.__get__(instance, owner)
            setattr(instance, cache_name, WorkflowStep(executable=AsyncFuncStepExecutable(func=bound_func)))
        return getattr(instance, cache_name)


class Workflow(Runnable[RunnableOutput], ABC):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self._input: list[AnyMessage] = []
        self._context: dict[str, Any] = {}

    @property
    def input(self) -> list[AnyMessage]:
        return self._input

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
        self._input = input
        self._context = RunContext.get().context

        # Builds out the execution graph
        # TODO The graph is reconstructed on each run?
        self._start_step: WorkflowStep = WorkflowStep(executable=EmptyStepExecutable("Start"))
        self.build(self._start_step)

        assert self._start_step is not None

        # Execute the workflow rooted at the start node
        queue: asyncio.Queue[Any] = asyncio.Queue()
        await queue.put(self._start_step)

        # Track running tasks
        tasks: set[asyncio.Task[Any]] = set()
        completed_steps: set[WorkflowStep] = set()

        async def execute_step(step: WorkflowStep) -> None:
            print("Executing:", step.name)

            # Get upstream results
            results = [u.result for u in step.upstream]

            await step.execute(*results)

            completed_steps.add(step)

            if step.requeue():
                await queue.put(step)
                return

            # Enqueue downstream steps
            for ds in step._downstream:
                await queue.put(ds)

        while not queue.empty() or tasks:
            # Drain current queue items into tasks
            while not queue.empty():
                step = await queue.get()
                task = asyncio.create_task(execute_step(step))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                queue.task_done()

            # Wait for all tasks on the queue to complete before proceeding
            # In certain cases this may not be optimal, but significantly simplifies implementation
            if tasks:
                done, _ = await asyncio.wait(tasks)
                # done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        return self.finalize()

    @abstractmethod
    def finalize(self) -> RunnableOutput:
        pass

    @abstractmethod
    def build(self, start: WorkflowStep) -> None:
        pass

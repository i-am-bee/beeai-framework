# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, TypeVar, Unpack, overload

from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry
from beeai_framework.workflows.v3.events import (
    workflow_v3_event_types,
)
from beeai_framework.workflows.v3.step import (
    EndWorkflowStep,
    FuncWorkflowStep,
    StartWorkflowStep,
    WorkflowBuilder,
    WorkflowStep,
)
from beeai_framework.workflows.v3.types import AsyncStepFunction

T = TypeVar("T", bound="Workflow")


class step:  # noqa: N801
    """
    Descriptor that turns an async method into a WorkflowStep.
    Users can refer to decorated methods and treat as WorkflowStep for composition.
    """

    def __init__(self, func: AsyncStepFunction, is_start: bool = False, is_end: bool = False) -> None:
        self.func = func
        self.name = func.__name__
        self.is_start = is_start
        self.is_end = is_end

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
            setattr(instance, cache_name, self.step_factory(func=bound_func))
        return getattr(instance, cache_name)

    def step_factory(self, func: AsyncStepFunction) -> WorkflowStep:
        return FuncWorkflowStep(func=func)

    # class start_step(step):  # noqa: N801
    #     def __init__(self, func: StartStepMethod) -> None:
    #         super().__init__(func, is_start=True)


class end_step(step):  # noqa: N801
    def step_factory(self, func: AsyncStepFunction) -> WorkflowStep:
        return EndWorkflowStep(func=func)


class Workflow(Runnable[RunnableOutput], ABC):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["workflow", "v3"], creator=self, events=workflow_v3_event_types)

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
        output: RunnableOutput = RunnableOutput(output=[])

        # Builds out the execution graph
        self._start_step: WorkflowStep = StartWorkflowStep(func=self.start)
        self.build(WorkflowBuilder([self._start_step]))

        # Execute the workflow rooted at the start node
        queue: asyncio.Queue[Any] = asyncio.Queue()
        await queue.put(self._start_step)

        # Track running tasks
        tasks: set[asyncio.Task[Any]] = set()
        completed_steps: set[WorkflowStep] = set()

        async def execute_step(step: WorkflowStep) -> None:
            # Send run input and kwargs to start step, otherwise get from upstream
            results = [input, kwargs] if isinstance(step, StartWorkflowStep) else [u.result for u in step.upstream]

            if not await step.condition(*results):
                # This assumes that this step/branch has been skipped
                # Allows downstream to proceed
                for ds in step._downstream:
                    ds.remaining_upstream -= 1
                return

            print("Executing:", step.name)

            await step.execute(*results)
            completed_steps.add(step)

            # Save output
            if isinstance(step, EndWorkflowStep):
                nonlocal output
                output = step.result

            if await step.requeue(*results):
                await queue.put(step)
                return

            # Queue any downstream if all upstream have completed
            # TODO: Loop until?
            for ds in step._downstream:
                ds.remaining_upstream -= 1
                if ds.remaining_upstream == 0:
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

        return output

    @abstractmethod
    def build(self, start: WorkflowBuilder) -> None:
        pass

    @abstractmethod
    async def start(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> Any:
        pass

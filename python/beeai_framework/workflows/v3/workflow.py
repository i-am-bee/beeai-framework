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
    FuncWorkflowStep,
    JoinWorkflowStep,
    StartWorkflowStep,
    WorkflowBuilder,
    WorkflowStep,
)
from beeai_framework.workflows.v3.types import AsyncStepFunction
from beeai_framework.workflows.v3.util import run_callable

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
            setattr(instance, cache_name, FuncWorkflowStep(func=bound_func))
        return getattr(instance, cache_name)


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
        # Builds out the execution graph
        self._start_step: WorkflowStep = StartWorkflowStep(func=self.start)
        self.build(WorkflowBuilder(root=self._start_step))

        # Execute the workflow rooted at the start node
        queue: asyncio.Queue[Any] = asyncio.Queue()
        await queue.put(self._start_step)

        # Track running tasks
        tasks: set[asyncio.Task[Any]] = set()
        completed_steps: list[WorkflowStep] = []

        async def execute_step(step: WorkflowStep) -> None:
            # Send run input and kwargs to start step, otherwise get from upstream

            results = []

            # TODO: Bind input
            if isinstance(step, StartWorkflowStep):
                results = [input, kwargs]
            else:
                for us in step.upstream_steps:
                    # Joins return aggregate
                    if isinstance(us, JoinWorkflowStep):
                        results.extend(us.result)
                    else:
                        results.append(us.result)

            print("Executing:", step.name)
            await step.execute(*results)
            completed_steps.append(step)

            if await step.requeue(*results):
                await queue.put(step)
                return

            step.has_executed = True

            # Enqueue downstream
            for ds_edge in step._downstream_edges:
                ds = ds_edge.target
                enqueue_ds = True
                # Check all upstream edges to determine execution conditions
                for up_edge in ds._upstream_edges:
                    up = up_edge.source

                    # If upstream has not executed and is its an and edge dont queue
                    if not up.has_executed and up_edge.type == "and":
                        enqueue_ds = False
                        break

                # Check for conditional execution
                if enqueue_ds and ds_edge.condition is not None:
                    if ds_edge.condition.fn:
                        enqueue_ds = bool(await run_callable(ds_edge.condition.fn, *results) == ds_edge.condition.key)
                    else:
                        enqueue_ds = bool(step.result == ds_edge.condition.key)

                if enqueue_ds:
                    ds.has_executed = False
                    await queue.put(ds)

        while not queue.empty() or tasks:
            # Drain current queue items into tasks
            while not queue.empty():
                step = await queue.get()
                task = asyncio.create_task(execute_step(step))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                queue.task_done()

            if tasks:
                # done, _ = await asyncio.wait(tasks)
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        output: RunnableOutput = (
            completed_steps[-1].result
            if completed_steps and isinstance(completed_steps[-1].result, RunnableOutput)
            else RunnableOutput(output=[])
        )
        return output

    @abstractmethod
    def build(self, start: WorkflowBuilder) -> None:
        pass

    @abstractmethod
    async def start(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> Any:
        pass

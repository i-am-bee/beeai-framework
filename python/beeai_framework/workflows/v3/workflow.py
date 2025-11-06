# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from functools import cached_property
from typing import Any, Unpack

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


class Workflow(Runnable[RunnableOutput]):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self._start_step: WorkflowStep | None = None

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["workflow", "v3"], creator=self, events=workflow_v3_event_types)

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:

        context = RunContext.get().context

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
                key = step.branch_fn(input, context)
                step_to_exec = step.next_steps[key]
            elif isinstance(step, WorkflowLoopUntil):
                step_to_exec = step.step

            assert step_to_exec is not None

            await step_to_exec._func(input, context)
            completed_steps.add(step_to_exec)

            if isinstance(step, WorkflowLoopUntil) and step.until_fn(input, context):
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
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        return RunnableOutput(output=[])

    def start(self, step: WorkflowStep) -> WorkflowStep:
        self._start_step = step
        return self._start_step

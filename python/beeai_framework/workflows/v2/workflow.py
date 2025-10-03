# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import time
from datetime import UTC, datetime
from functools import cached_property
from itertools import zip_longest
from typing import Any, Unpack

from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import RunContext, RunMiddlewareType
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableInput
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry
from beeai_framework.workflows.v2.events import (
    StartWorkflowEvent,
    StartWorkflowStepEvent,
    WorkflowEventNames,
    workflow_event_types,
)
from beeai_framework.workflows.v2.step import WorkflowStep, WorkflowStepExecution
from beeai_framework.workflows.v2.types import AsyncFunc


class Workflow(Runnable[RunnableOutput]):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)
        self._input: list[AnyMessage] = []
        self._output: list[AnyMessage] = []
        self._steps: dict[str, WorkflowStep] = {}
        self._start_step: WorkflowStep | None = None
        self._end_steps: set[WorkflowStep] = set()
        self._running_tasks: set[asyncio.Task[Any]] = set()
        self._queue: asyncio.Queue[WorkflowStep] = asyncio.Queue()
        self._scan()

    def add_running_task(self, task: asyncio.Task[Any]) -> None:
        self._running_tasks.add(task)

    def inspect(self, step: AsyncFunc | str) -> WorkflowStep:
        key = step if isinstance(step, str) else step.__name__
        return self._steps[key]

    def _scan(self) -> None:
        """Scan decorated methods to build execution graph"""
        methods = inspect.getmembers(self, inspect.ismethod)

        # Create all steps first
        for name, method in methods:
            if hasattr(method, "_is_step"):
                is_start = hasattr(method, "_is_start") and method._is_start
                is_end = hasattr(method, "_is_end") and method._is_end
                is_fork = hasattr(method, "_is_fork") and method._is_fork
                is_join = hasattr(method, "_is_join") and method._is_join
                retries = hasattr(method, "_retries")

                step = WorkflowStep(method, start=is_start, end=is_end, fork=is_fork, join=is_join, retries=retries)
                self._steps[name] = step

                if step.is_start:
                    self._start_step = step

                if step.is_end:
                    self._end_steps.add(step)

            # TODO multiple predicate decorators
            if hasattr(method, "_when_predicate"):
                step.add_predicate(method._when_predicate)

        # Once all steps have been created build dependency graph
        for name, method in methods:
            if hasattr(method, "_dependencies") and hasattr(method, "_dependency_type"):
                dependency_type = method._dependency_type
                dependencies = method._dependencies
                self._steps[name].type = dependency_type

                for dep in dependencies:
                    if isinstance(dep, str):
                        dep = dict(methods).get(dep)

                    self._steps[name].add_dependency(self._steps[dep.__name__])

    async def _run(self) -> None:
        run_context = RunContext.get()

        await run_context.emitter.emit(
            WorkflowEventNames.START_WORKFLOW,
            StartWorkflowEvent(),
        )

        assert self._start_step is not None
        await self._queue.put(self._start_step)

        while not self._queue.empty() or self._running_tasks:
            # Start all tasks in queue concurrently
            while not self._queue.empty():
                step = await self._queue.get()
                task = asyncio.create_task(self._run_step(step))
                self._running_tasks.add(task)
                task.add_done_callback(self._running_tasks.discard)

            if self._running_tasks:
                # Wait until any task completes
                await asyncio.wait(self._running_tasks, return_when=asyncio.FIRST_COMPLETED)

    async def _run_step(self, step: WorkflowStep) -> None:
        # If predicates exists and any one is false then bail
        if step.predicates and any(p(self, *step.inputs) for p in step.predicates) is False:
            return

        started_at = datetime.now(UTC)
        start_perf = time.perf_counter()

        await RunContext.get().emitter.emit(
            WorkflowEventNames.START_WORKFLOW_STEP,
            StartWorkflowStepEvent(step=step),
        )
        if step.is_start:
            result = await Retryable(
                RetryableInput(
                    executor=lambda _: step.func(self._input),
                    config=RetryableConfig(max_retries=step.retries),
                )
            ).get()
        else:
            if step.is_fork:
                safe_params = [i if i is not None else [] for i in step.inputs]
                params = list(zip_longest(*safe_params, fillvalue=None))
                tasks = [step.func(*p) for p in params]

                result = await asyncio.gather(*tasks)
            elif step.is_join:
                # Include the inputs to the original fork
                fork_inputs = [d.inputs for d in step.dependencies]
                flat_fork_input = [item for sublist in fork_inputs for item in sublist]
                inputs = flat_fork_input + step.inputs

                result = await Retryable(
                    RetryableInput(
                        executor=lambda _: step.func(*inputs),
                        config=RetryableConfig(max_retries=step.retries),
                    )
                ).get()
            else:
                result = await Retryable(
                    RetryableInput(
                        executor=lambda _: step.func(*step.inputs),
                        config=RetryableConfig(max_retries=step.retries),
                    )
                ).get()

        step.executions.append(
            WorkflowStepExecution(
                inputs=tuple(step.inputs),
                output=result,
                error=None,
                started_at=started_at,
                ended_at=datetime.now(UTC),
                duration=time.perf_counter() - start_perf,
            )
        )

        if step._is_end:
            self._output = result

        # Enqueue dependents (that are waiting on the completion of this step)
        for dep in step.dependents:
            # Insert this result into the inputs at correct index
            idx = dep.dependencies.index(step)
            # Set the input at the correct dependency index
            dep.inputs[idx] = result

            if dep.type == "AND":
                dep.completed_dependencies.append(step)
                if set(dep.completed_dependencies) == set(dep.dependencies):
                    await self._queue.put(dep)  # All dependencies are done, queue dependent
            elif dep.type == "OR":
                await self._queue.put(dep)  # Can queue immediately for OR

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["workflow", "v2"], creator=self, events=workflow_event_types)

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
        self._input = input
        await self._run()
        return RunnableOutput(output=self._output)

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from collections.abc import Callable
from itertools import zip_longest
from typing import Any

from beeai_framework.backend.message import AnyMessage
from beeai_framework.workflows.v2.types import AsyncFunc, DependencyType


class WorkflowStep:
    def __init__(self, func: AsyncFunc) -> None:
        self.func = func
        self.name = func.__name__
        self._is_start = False
        self._is_fork = False
        self._is_join = False
        self.forked: list[WorkflowStep] = []
        self.dependencies: list[WorkflowStep] = []
        self.dependents: list[WorkflowStep] = []
        self.completed_dependencies: list[WorkflowStep] = []
        self.inputs: list[Any | None] = []

        self.type: DependencyType = "AND"
        self.last_result: Any = None
        self.iterations = 0
        self.max_iterations = 10
        self.completed_event = asyncio.Event()
        self.predicate: Callable[..., bool] | None = None

    def add_dependency(self, dep: "WorkflowStep") -> None:
        self.dependencies.append(dep)
        self.inputs.append(None)

    def add_dependent(self, dep: "WorkflowStep") -> None:
        self.dependents.append(dep)


class Workflow:
    def __init__(self) -> None:
        self._messages: list[AnyMessage] = []
        self._start_step: WorkflowStep | None = None
        self._end_step: WorkflowStep | None = None
        self._steps: dict[str, WorkflowStep] = {}
        self.running_tasks: set[asyncio.Task[Any]] = set()
        self.queue: asyncio.Queue[WorkflowStep] = asyncio.Queue()
        self._scan()

    def _scan(self) -> None:
        methods = inspect.getmembers(self, inspect.ismethod)

        for name, method in methods:
            if hasattr(method, "_is_step"):
                step = WorkflowStep(method)
                self._steps[name] = step
                if hasattr(method, "_is_start") and method._is_start:
                    self._start_step = step
                    self._start_step._is_start = True

        for name, method in methods:
            if hasattr(method, "_is_fork"):
                self._steps[name]._is_fork = method._is_fork

            if hasattr(method, "_is_join"):
                self._steps[name]._is_join = method._is_join

            if hasattr(method, "_when_predicate"):
                self._steps[name].predicate = method._when_predicate

            if hasattr(method, "_dependencies") and hasattr(method, "_dependency_type"):
                dependency_type = method._dependency_type
                dependencies = method._dependencies
                self._steps[name].type = dependency_type

                for dep in dependencies:
                    if isinstance(dep, str):
                        dep = dict(methods).get(dep)

                    self._steps[name].add_dependency(self._steps[dep.__name__])
                    self._steps[dep.__name__].add_dependent(self._steps[name])

    async def _run(self) -> None:
        assert self._start_step is not None
        await self.queue.put(self._start_step)
        # else:
        #     # Otherwise add all non dependent steps
        #     for _, step in self._steps.items():
        #         if not step.dependencies:
        #             await self.queue.put(step)

        while not self.queue.empty() or self.running_tasks:
            # Start all tasks in queue concurrently
            while not self.queue.empty():
                step = await self.queue.get()
                if step.iterations >= step.max_iterations:
                    continue
                task = asyncio.create_task(self._run_step(step))
                self.running_tasks.add(task)
                task.add_done_callback(self.running_tasks.discard)

            if self.running_tasks:
                # Wait until any task completes
                await asyncio.wait(self.running_tasks, return_when=asyncio.FIRST_COMPLETED)

    async def _run_step(self, step: WorkflowStep) -> None:
        # If a predicate exists and it is false then bail
        if step.predicate and step.predicate(self, *step.inputs) is False:
            return

        if step._is_start:
            result = await step.func(self._messages)
        else:
            if step._is_fork:
                safe_params = [i if i is not None else [] for i in step.inputs]
                params = list(zip_longest(*safe_params, fillvalue=None))
                tasks = [step.func(*p) for p in params]
                result = await asyncio.gather(*tasks)
            elif step._is_join:
                # Include the inputs to the original fork
                fork_inputs = [d.inputs for d in step.dependencies]
                flat_fork_input = [item for sublist in fork_inputs for item in sublist]
                inputs = flat_fork_input + step.inputs
                result = await step.func(*inputs)
            else:
                result = await step.func(*step.inputs)

        step.last_result = result
        step.iterations += 1

        # Enqueue dependents (that are waiting on the completion of this step)
        for dep in step.dependents:
            # Insert this result into the inputs at correct index
            idx = dep.dependencies.index(step)
            dep.inputs[idx] = result

            if dep.type == "AND":
                dep.completed_dependencies.append(step)
                if set(dep.completed_dependencies) == set(dep.dependencies):
                    await self.queue.put(dep)  # All dependencies are done, proceed
            elif dep.type == "OR":
                await self.queue.put(dep)  # Can proceed without checks for OR

    async def run(self, input: list[AnyMessage]) -> list[AnyMessage]:
        self._messages = input
        await self._run()
        return self._messages

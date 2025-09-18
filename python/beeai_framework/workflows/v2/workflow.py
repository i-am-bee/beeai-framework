# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from typing import Any

from beeai_framework.backend.message import AnyMessage
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.workflow_types import AsyncFunc, DependencyType


class Step:
    def __init__(self, func: AsyncFunc) -> None:
        self.func = func
        self.name = func.__name__
        self.dependencies: set[Step] = set()
        self.dependents: set[Step] = set()
        self.completed_dependencies: set[Step] = set()
        self.type: DependencyType = "AND"
        self.last_result: Any = None
        self.iterations = 0
        self.max_iterations = 2
        self.completed_event = asyncio.Event()

    def add_dependency(self, dep: "Step") -> None:
        self.dependencies.add(dep)

    def add_dependent(self, dep: "Step") -> None:
        self.dependents.add(dep)


class Workflow:
    def __init__(self) -> None:
        self._start: str | None = None
        self._graph: dict[str, Step] = {}
        self.running_tasks: set[asyncio.Task[Any]] = set()
        self.queue: asyncio.Queue[Step] = asyncio.Queue()
        self._scan()

    def _scan(self) -> None:

        methods = inspect.getmembers(self, inspect.ismethod)

        for name, method in methods:
            if hasattr(method, "_is_step"):
                self._graph[name] = Step(method)

            if hasattr(method, "_is_start") and method._is_start:
                self._start = name

        for name, method in methods:
            if hasattr(method, "_dependencies") and hasattr(method, "_dependency_type"):
                dependency_type = method._dependency_type
                dependencies = method._dependencies

                self._graph[name].type = dependency_type

                for dep in dependencies:
                    if isinstance(dep, str):
                        dep = dict(methods).get(dep)

                    self._graph[name].add_dependency(self._graph[dep.__name__])
                    self._graph[dep.__name__].add_dependent(self._graph[name])

    async def _run(self) -> None:

        steps = self._graph.values()

        for step in steps:
            if not step.dependencies:
                await self.queue.put(step)

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

    async def _run_step(self, step: Step) -> None:

        result = await step.func()
        step.last_result = result
        step.iterations += 1

        # Enqueue dependents reactively
        for dep in step.dependents:
            if dep.type == "AND":
                dep.completed_dependencies.add(step)
                if dep.completed_dependencies == dep.dependencies:
                    await self.queue.put(dep)
            elif dep.type == "OR":
                await self.queue.put(dep)

    async def run(self, messages: list[AnyMessage]) -> None:
        await self._run()


# Async main function
async def main() -> None:

    # class MyWorkflow(Workflow):
    #     @start
    #     async def start(self) -> None:
    #         print("Start!!!")

    #     @after(start)
    #     async def a(self) -> None:
    #         print("A runs after start")

    #     @after(start)
    #     async def b(self) -> None:
    #         print("B runs after start")

    #     @after(a, b)
    #     async def end(self) -> None:
    #         print("End after A and B are done!")

    # flow = MyWorkflow()
    # await flow.run([])

    class MyWorkflow(Workflow):

        def __init__(self) -> None:
            self.value = 0
            super().__init__()

        @start
        async def start(self) -> None:
            print("Start")

        @after(start, "act", "think", type="OR")
        async def think(self) -> None:
            await asyncio.sleep(3)
            print("Think")

        @after(think)
        async def act(self) -> None:
            print("Act")

        @after(act)
        async def observe(self) -> None:
            print("Observe")

    flow = MyWorkflow()
    await flow.run([])


# Entry point
if __name__ == "__main__":
    asyncio.run(main())

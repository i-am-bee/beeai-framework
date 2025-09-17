# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from beeai_framework.backend.message import AnyMessage
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.types import AsyncFunc


class Step:
    def __init__(self, func: AsyncFunc) -> None:
        self.func = func
        self.name = func.__name__
        self._dependencies: list[Step] = []
        self.completed: asyncio.Event = asyncio.Event()

    def add_dependency(self, dep: "Step") -> None:
        self._dependencies.append(dep)

    @property
    def dependencies(self) -> list["Step"]:
        return self._dependencies


class Workflow:
    def __init__(self) -> None:
        self._start: str | None = None
        self._graph: dict[str, Step] = {}
        self._scan()

    def _scan(self) -> None:

        for name in dir(self):
            attr = getattr(self, name)
            if callable(attr):
                _func = getattr(attr, "_func", None)
                if _func:
                    self._graph[_func.__name__] = Step(_func)

        for name in dir(self):
            attr = getattr(self, name)
            if callable(attr):
                _func = getattr(attr, "_func", None)
                if _func:
                    if hasattr(attr, "_is_start") and attr._is_start:
                        self._start = _func.__name__
                    elif hasattr(attr, "_dependencies"):
                        dependencies = getattr(attr, "_dependencies", [])
                        for dep in dependencies:
                            self._graph[_func.__name__].add_dependency(self._graph[dep.__name__])

    async def _run(self, step: Step) -> None:
        # Wait for all dependencies to complete
        for dep in step.dependencies:
            await dep.completed.wait()

        print(f"Running {step.name}")
        await step.func(self)
        step.completed.set()  # signal dependents

    async def run(self, messages: list[AnyMessage]) -> None:
        if self._graph:
            """run all non dependent methods to start"""
            coros = [self._run(step) for step in self._graph.values()]
            await asyncio.gather(*coros)


# Async main function
async def main() -> None:

    class MyWorkflow(Workflow):
        @start
        async def start(self) -> None:
            print("Start!!!")

        @after(start)
        async def a(self) -> None:
            print("A runs after start")

        @after(start)
        async def b(self) -> None:
            print("B runs after start")

        @after(a, b)
        async def end(self) -> None:
            print("End after A and B are done!")

    flow = MyWorkflow()
    await flow.run([])


# Entry point
if __name__ == "__main__":
    asyncio.run(main())

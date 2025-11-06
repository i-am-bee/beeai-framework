# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import time
from datetime import UTC, datetime
from functools import cached_property
from itertools import zip_longest
from pathlib import Path
from typing import Any, Unpack

from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import RunContext, RunMiddlewareType
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry
from beeai_framework.workflows.v2.events import (
    WorkflowStartEvent,
    WorkflowStartStepEvent,
    workflow_v2_event_types,
)
from beeai_framework.workflows.v2.step import WorkflowStep, WorkflowStepExecution
from beeai_framework.workflows.v2.types import AsyncMethod, AsyncMethodSet
from beeai_framework.workflows.v2.util import prepare_args, run_callable


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

    def print_html(self, path: Path | str | None = None) -> None:
        def to_mermaid(direction: str = "TD") -> list[str]:
            lines = [f"flowchart-elk {direction}"]
            visited = set()

            def dfs(step: WorkflowStep) -> None:
                if step in visited:
                    return
                visited.add(step)
                dependents = step.dependents
                for dep in dependents:
                    lines.append(f"\t_{step.name}({step.name}) --> _{dep.name}({dep.name})")
                    dfs(dep)
                if not dependents:
                    lines.append(f"\t_{step.name}({step.name})")

            if self._start_step:
                dfs(self._start_step)

            return lines

        mermaid_code_list = to_mermaid()
        mermaid_code_list.append("classDef _cls_start fill:#ffe5e5,stroke:#d32f2f,color:#b71c1c")
        mermaid_code_list.append("classDef _cls_end fill:#e0f7fa,stroke:#00796B,color:#004d40")

        if self._start_step:
            mermaid_code_list.append(f"class _{self._start_step.name} _cls_start")

        for step in self._end_steps:
            mermaid_code_list.append(f"class _{step.name} _cls_end")

        mermaid_code = "\n".join(mermaid_code_list)
        default_filename = f"{self.__class__.__name__.lower()}.html"

        # If no path provided, write next to current module file
        if path is None:
            file_path = Path(__file__).parent / default_filename
        else:
            path = Path(path)
            # If path is a directory, append default filename
            file_path = path / default_filename if path.is_dir() or not path.suffix else path

        file_path.parent.mkdir(parents=True, exist_ok=True)

        html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Mermaid Diagram</title>
                <style>
                    html, body {{
                        height: 100%;
                        margin: 0;
                    }}

                    body {{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }}

                    .diagram-container {{
                        text-align: center;
                        width: 100%;
                    }}
                </style>
            </head>
            <body>
                <div class="diagram-container">
                    <pre class="mermaid">
                        {mermaid_code}
                    </pre>
                </div>

                <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
                    import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs';
                    mermaid.registerLayoutLoaders(elkLayouts);
                    mermaid.initialize({{startOnLoad: true}});
                </script>
            </body>
            </html>
            """

        file_path.write_text(html_template, encoding="utf-8")

    def inspect(self, step: AsyncMethod | str) -> WorkflowStep:
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
                retries = method._retries if hasattr(method, "_retries") else 0

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
            if hasattr(method, "_dependency"):
                dependency = method._dependency
                if isinstance(dependency, str):
                    m = dict(methods).get(dependency)
                    if m is not None:
                        self._steps[name].add_dependency(self._steps[m.__name__])
                elif isinstance(dependency, AsyncMethodSet):
                    for method_name in dependency.methods:
                        m = dict(methods).get(method_name)
                        if m is not None:
                            self._steps[name].add_dependency(self._steps[m.__name__])
                    self._steps[name].condition = dependency.condition
                elif callable(dependency):
                    self._steps[name].add_dependency(self._steps[dependency.__name__])

    async def _run(self) -> None:
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
                done, _ = await asyncio.wait(self._running_tasks, return_when=asyncio.FIRST_COMPLETED)

                # Raise exceptions for any finished tasks
                for task in done:
                    exception = task.exception()
                    if exception:
                        raise exception  # re-raise the exception from the task

    async def _run_step(self, step: WorkflowStep) -> None:
        step_inputs: list[Any] = (
            [self._input] if step.is_start else [step.inputs[k] for k in sorted(step.inputs.keys())]
        )

        for p in step.predicates:
            args = prepare_args(p, self, *step_inputs)
            p_res = await run_callable(p, *args)
            if not p_res:
                return

        started_at = datetime.now(UTC)
        start_perf = time.perf_counter()

        await RunContext.get().emitter.emit(
            "start_step",
            WorkflowStartStepEvent(step=step),
        )

        if step.is_start:
            result = await run_callable(step.func, *step_inputs)
        else:
            if step.is_fork:
                safe_params = [i if i is not None else [] for i in step_inputs]
                params = list(zip_longest(*safe_params, fillvalue=None))
                tasks = [run_callable(step.func, *p) for p in params]
                # TODO Retry forked
                result = await asyncio.gather(*tasks)
            elif step.is_join:
                # Include the inputs to the original fork
                fork_inputs = [[d.inputs[k] for k in sorted(d.inputs.keys())] for d in step.dependencies]
                flat_fork_input = [item for sublist in fork_inputs for item in sublist]
                inputs = flat_fork_input + step_inputs
                result = await run_callable(step.func, *inputs)
            else:
                result = await run_callable(step.func, *step_inputs)

        step.executions.append(
            WorkflowStepExecution(
                inputs=step_inputs,
                output=result,
                error=None,
                started_at=started_at,
                ended_at=datetime.now(UTC),
                duration=time.perf_counter() - start_perf,
            )
        )

        if step._is_end:
            self._output = result or []

        # Enqueue dependents (that are waiting on the completion of this step)
        for dep in step.dependents:
            # Insert this result into the inputs at correct index
            idx = dep.dependencies.index(step)

            # Set the input at the correct dependency index
            if dep.condition == "and":
                dep.inputs[idx] = result
            else:
                dep.inputs[0] = result

            if dep.condition == "and":
                dep.completed_dependencies.append(step)
                if set(dep.completed_dependencies) == set(dep.dependencies):
                    await self._queue.put(dep)  # All dependencies are done, queue dependent
            elif dep.condition == "or":
                await self._queue.put(dep)  # Can queue immediately for OR

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["workflow", "v2"], creator=self, events=workflow_v2_event_types)

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
        run_context = RunContext.get()

        await run_context.emitter.emit(
            "start",
            WorkflowStartEvent(),
        )

        self._input = input
        await self._run()
        return RunnableOutput(output=self._output)

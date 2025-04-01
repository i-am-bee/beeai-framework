# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import inspect
from functools import cached_property
from typing import ClassVar, Final, Generic, Literal

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.agents.types import AgentMetaDetail
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.utils.models import ModelLike, check_model, to_model, to_model_optional
from beeai_framework.utils.strings import to_safe_word
from beeai_framework.workflows.errors import WorkflowError
from beeai_framework.workflows.events import WorkflowErrorEvent, WorkflowStartEvent, WorkflowSuccessEvent
from beeai_framework.workflows.types import (
    WorkflowHandler,
    WorkflowRun,
    WorkflowRunOptions,
    WorkflowState,
    WorkflowStepDefinition,
    WorkflowStepRes,
)

T = TypeVar("T", bound=BaseModel)
K = TypeVar("K", default=str)


class Workflow(Generic[T, K]):
    START: Final[Literal["__start__"]] = "__start__"
    SELF: Final[Literal["__self__"]] = "__self__"
    PREV: Final[Literal["__prev__"]] = "__prev__"
    NEXT: Final[Literal["__next__"]] = "__next__"
    END: Final[Literal["__end__"]] = "__end__"

    _RESERVED_STEP_NAMES: ClassVar = [START, SELF, PREV, NEXT, END]

    def __init__(self, schema: type[T], name: str = "Workflow") -> None:
        self._name = name
        self._schema = schema
        self._steps: dict[K, WorkflowStepDefinition[T, K]] = {}
        self._start_step: K | None = None

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["workflow", to_safe_word(self._name)],
            creator=self,
            events={
                "start": WorkflowStartEvent[T, K],
                "success": WorkflowSuccessEvent[T, K],
                "error": WorkflowErrorEvent[T, K],
            },
        )

    @property
    def steps(self) -> dict[K, WorkflowStepDefinition[T, K]]:
        return self._steps

    @property
    def step_names(self) -> list[K]:
        return list(self.steps.keys())

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> type[T]:
        return self._schema

    @property
    def start_step(self) -> K | None:
        return self._start_step

    def add_step(
        self, step_name: K, runnable: WorkflowHandler[T, K], agent_metadata: AgentMetaDetail | None = None
    ) -> "Workflow[T, K]":
        if (len(str(step_name).strip())) == 0:
            raise ValueError("Step name cannot be empty!")

        if step_name in self.steps:
            raise ValueError(f"The name '{step_name}' has already been used!")

        if step_name in Workflow._RESERVED_STEP_NAMES:
            raise ValueError(f"The name '{step_name}' is reserved and cannot be used!")

        self.steps[step_name] = WorkflowStepDefinition[T, K](handler=runnable, agent_metadata=agent_metadata)

        return self

    def delete_step(self, step_name: K) -> "Workflow[T, K]":
        if step_name not in self.steps:
            raise WorkflowError(f"Step '${step_name}' was not found.")

        del self.steps[step_name]

        if self.start_step == step_name:
            self._start_step = None

        return self

    def set_start(self, name: K) -> "Workflow[T, K]":
        self._start_step = name
        return self

    def run(
        self, state: ModelLike[T], options: ModelLike[WorkflowRunOptions[K]] | None = None
    ) -> Run[WorkflowRun[T, K]]:
        options = to_model_optional(WorkflowRunOptions[K], options)

        async def handler(context: RunContext) -> WorkflowRun[T, K]:
            run = WorkflowRun[T, K](state=to_model(self._schema, state))
            # handlers = WorkflowRunContext(steps=run.steps, signal=context.signal, abort=lambda r: context.abort(r))
            next = self._find_step(self.start_step or self.step_names[0]).current or Workflow.END

            while next and next != Workflow.END:
                step = self.steps.get(next)
                if step is None:
                    raise WorkflowError(f"Step '{next}' was not found.")

                await context.emitter.emit("start", WorkflowStartEvent[T, K](run=run, step=next))

                try:
                    step_res = WorkflowStepRes[T, K](name=next, state=run.state.model_copy(deep=True))
                    run.steps.append(step_res)

                    if inspect.iscoroutinefunction(step.handler):
                        step_next = await step.handler(step_res.state)  # , handlers)
                    else:
                        step_next = await asyncio.to_thread(step.handler, step_res.state)  # handlers)

                    check_model(step_res.state)
                    run.state = step_res.state

                    # Route to next step
                    if step_next == Workflow.START:
                        next = run.steps[0].name
                    elif step_next == Workflow.PREV:
                        next = self._find_step(next).prev or Workflow.END
                    elif step_next == Workflow.SELF:
                        next = self._find_step(next).current
                    elif step_next is None or step_next == Workflow.NEXT:
                        next = self._find_step(next).next or Workflow.END
                    else:
                        next = step_next

                    await context.emitter.emit(
                        "success",
                        WorkflowSuccessEvent[T, K](
                            run=run.model_copy(),
                            state=run.state,
                            step=step_res.name,
                            next=next,
                        ),
                    )
                except Exception as e:
                    err = FrameworkError.ensure(e)
                    await context.emitter.emit(
                        "error",
                        WorkflowErrorEvent[T, K](
                            run=run.model_copy(),
                            step=next,
                            error=err,
                        ),
                    )
                    raise err

            if run.state:  # TODO: add output schema
                run.result = run.state.model_copy()

            return run

        return RunContext.enter(
            self,
            handler,
            signal=options.signal if options else None,
            run_params={"state": state, "options": options},
        )

    def _find_step(self, current: K) -> WorkflowState[K]:
        index = self.step_names.index(current)
        return WorkflowState[K](
            prev=self.step_names[index - 1] if 0 <= index - 1 < len(self.step_names) else None,
            current=self.step_names[index],
            next=self.step_names[index + 1] if 0 <= index + 1 < len(self.step_names) else None,
        )

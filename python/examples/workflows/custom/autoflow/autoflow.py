"""
The following example illustrates the concept of AutoFlow, where each step is represented
by an arbitrary function, and the entire execution process is orchestrated by an LLM acting as an intelligent agent.
After each step invocation the router (LLM) is called to determine the next step.
"""

import asyncio
import inspect
import json
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Final, Literal

from dotenv import load_dotenv

from beeai_framework.backend import AssistantMessage, ChatModel, SystemMessage, UserMessage
from beeai_framework.context import Run
from beeai_framework.emitter import Emitter, EventMeta
from beeai_framework.logger import Logger
from beeai_framework.memory import BaseMemory, UnconstrainedMemory
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import tool as create_tool
from beeai_framework.tools.weather import OpenMeteoTool, OpenMeteoToolInput
from beeai_framework.utils.asynchronous import ensure_async
from beeai_framework.utils.strings import to_json
from beeai_framework.workflows import Workflow, WorkflowRun, WorkflowSuccessEvent
from examples.workflows.custom.autoflow.prompts import (
    AutoflowReasonPrompt,
    AutoflowReasonPromptInput,
    AutoflowResponsePrompt,
    AutoflowResponsePromptInput,
)
from examples.workflows.custom.autoflow.types import (
    AutoflowHandler,
    AutoflowState,
)

load_dotenv()


logger = Logger(__name__)

_storage = ContextVar[AutoflowState]("_storage")


class Autoflow:
    ROUTER: Final[Literal["__router__"]] = "__router__"
    FINAL_ANSWER: Final[Literal["final_answer"]] = "final_answer"

    def __init__(
        self,
        name: str = "dynamic",
        *,
        model: ChatModel,
        memory: BaseMemory,
        router_reason_template: PromptTemplate[AutoflowReasonPromptInput] | None = None,
        router_response_template: PromptTemplate[AutoflowResponsePromptInput] | None = None,
        save_intermediate_steps: bool = False,
        allow_immediate_step_reuse: bool = True,
    ) -> None:
        self._model = model
        self._memory = memory
        self._workflow = Workflow(name=name, schema=AutoflowState)
        self._allow_immediate_step_reuse = allow_immediate_step_reuse

        self._workflow.add_step(Autoflow.ROUTER, self._route)
        self._handler_by_name: dict[str, AutoflowHandler] = {}
        self._router_reason_template = router_reason_template or AutoflowReasonPrompt
        self._router_response_template = router_response_template or AutoflowResponsePrompt
        self._transition_overrides: dict[str, str] = {}
        self._save_intermediate_steps = save_intermediate_steps

        @self.add_step(name=Autoflow.FINAL_ANSWER, next_step=Workflow.END)
        async def final_answer(answer: str) -> str:
            """Provides the final answer to the user's message (task). This must be the last called action."""

            return answer

    @staticmethod
    def get_context() -> AutoflowState:
        return _storage.get()

    @property
    def emitter(self) -> Emitter:
        return self._workflow.emitter

    def add_step(
        self,
        name: str | None = None,
        next_step: str | None = None,
    ) -> Callable[..., "Autoflow"]:
        def register(fn: AutoflowHandler) -> "Autoflow":
            final_name = name or fn.__name__
            if not final_name:
                raise ValueError("Step function must have a name.")

            if not inspect.getdoc(fn):
                raise ValueError("Step function must have a docstring.")

            self._workflow.add_step(final_name, self._execute)
            self._handler_by_name[final_name] = fn

            if next_step is not None:
                self._transition_overrides[final_name] = next_step

            return self

        return register

    def del_step(self, name: str) -> None:
        self._workflow.delete_step(name)
        self._transition_overrides.pop(name, None)
        self._handler_by_name.pop(name, None)

    async def _execute(self, state: AutoflowState) -> str:
        assert state.last_input is not None, "No input provided."
        assert state.next_step, "No next step provided."
        state.last_step = state.next_step
        state.next_step = self._transition_overrides.get(state.last_step, Autoflow.ROUTER)

        fn = self._handler_by_name[state.last_step]
        state.last_result = await ensure_async(fn)(**state.last_input)  # type: ignore
        await state.memory.add(
            AssistantMessage(
                self._router_response_template.render(
                    AutoflowResponsePromptInput(
                        step=state.last_step,
                        input=to_json(state.last_input),
                        output=str(state.last_result),
                    )
                )
            )
        )
        return state.next_step

    async def _route(self, state: AutoflowState) -> str:
        tools = [
            create_tool(handler, name=name)
            for name, handler in self._handler_by_name.items()
            if not name.startswith("__") and (self._allow_immediate_step_reuse or name != state.last_step)
        ]
        response = await self._model.run(
            [
                SystemMessage(
                    self._router_reason_template.render(
                        AutoflowReasonPromptInput(
                            actions=[{"name": tool.name, "description": tool.description} for tool in tools],
                            context=state.context,
                        )
                    )
                ),
                *state.memory.messages,
            ],
            tool_choice="required" if len(tools) > 1 else tools[0],
            tools=tools,
        )
        tool_call = response.get_tool_calls()[0]
        state.next_step = tool_call.tool_name
        state.last_input = json.loads(tool_call.args)
        return state.next_step

    def reset(self) -> None:
        self._memory.reset()

    def run(
        self, task: str, *, context: str | None = None, start_step: str | None = None
    ) -> Run[WorkflowRun[AutoflowState, str]]:
        state = AutoflowState(
            task=task,
            context=context or "",
            memory=self._memory,
            last_result=None,
            last_input={},
            last_step=None,
            next_step=start_step or Autoflow.ROUTER,
        )

        async def start(data: Any, event: EventMeta) -> None:
            await state.memory.add(UserMessage(state.task))

        async def stop(data: WorkflowRun[AutoflowState, str], event: EventMeta) -> None:
            if self._save_intermediate_steps:
                self._memory.reset()
                await self._memory.add_many(data.state.memory.messages)
            else:
                await self._memory.add(AssistantMessage(data.state.last_result))

        return (
            self._workflow.set_start(state.next_step)
            .run(state)
            .on(
                lambda event: event.path == ".".join(["run", *self._workflow.emitter.namespace, "start"]),
                start,
            )
            .on(
                lambda event: event.path == ".".join(["run", *self._workflow.emitter.namespace, "success"]),
                stop,
            )
        )


async def main() -> None:
    model = ChatModel.from_name("openai:gpt-4.1-mini")
    workflow = Autoflow(model=model, memory=UnconstrainedMemory(), allow_immediate_step_reuse=True)

    @workflow.add_step()
    async def weather_forecast(location: str, date: str) -> str:
        """Retrieves the weather forecast for a given location and date. Date must be in the format YYYY-MM-DD."""

        tool = OpenMeteoTool()
        date_f = datetime.strptime(date, "%Y-%m-%d").date()  # noqa: DTZ007
        result = await tool.run(OpenMeteoToolInput(location_name=location, start_date=date_f, end_date=date_f))
        formatted_result = await model.run(
            [
                SystemMessage(
                    "You are a weather forecasting assistant.\n"
                    "Your task is to briefly summarize the weather forecast from the raw data provided in the user message."
                    "Your response should not contain any additional comments."
                ),
                UserMessage(f"Location: ${location}"),
                UserMessage(result.get_text_content()),
            ]
        )
        return formatted_result.get_text_content()

    @workflow.add_step()
    def get_current_date_and_time() -> str:
        """Retrieves the current date and time."""

        return f"Today is {datetime.now(tz=UTC).strftime('%A, %B %-d, %Y at %-I:%M %p UTC')}."

    # @workflow.add_step()
    # def example_function() -> str:
    #     """Description of your function"""
    #
    #     state = Autoflow.get_context()
    #     print(state.last_input)
    #     print(state.task)
    #     print(state.context)
    #     return "..."

    def log_intermediate_steps(data: WorkflowSuccessEvent[AutoflowState], event: EventMeta) -> None:
        if data.next in [Autoflow.ROUTER, Workflow.END] and data.step != data.next:
            print(" Output: ", data.state.last_result, "\n")
            print(f"{data.step} -> {data.next}")
        else:
            print(f"{data.step} -> {data.next}")
            print(" Input: ", data.state.last_input)

    response = await workflow.run(  # noqa: F841
        task="Create a detailed travel plan for my upcoming trip to Prague next weekend. The plan should include a daily schedule along with the weather forecast for each day.",
        start_step="get_current_date_and_time",  # optional
    ).on("success", log_intermediate_steps)

    # Additional data
    # print(response.state.last_result)
    # print(response.state.memory)
    # print(response.state.last_step)
    # print(response.state.last_input)
    # print(response.state.next_step)


if __name__ == "__main__":
    asyncio.run(main())

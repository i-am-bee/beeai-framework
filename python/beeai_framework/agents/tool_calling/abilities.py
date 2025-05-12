# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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
import math
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Self

from pydantic import BaseModel, Field, create_model

from beeai_framework.agents.tool_calling.types import (
    AgentAbility,
    AgentAbilityState,
    AnyAbility,
    TAbilityInput,
    ToolCallingAgentRunState,
)
from beeai_framework.backend import AssistantMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.tools import AnyTool, StringToolOutput, Tool, ToolRunOptions
from beeai_framework.utils import MaybeAsync

if TYPE_CHECKING:
    from beeai_framework.agents.tool_calling.agent import ToolCallingAgent


class FinalAnswerAbility(AgentAbility[BaseModel]):
    def __init__(
        self, expected_output: str | type[BaseModel] | None, state: ToolCallingAgentRunState, double_check: bool = False
    ) -> None:  # TODO: propagate state with tool context..
        super().__init__()
        self.name = "final_answer"
        self.description = "Sends the final answer to the user"
        self._expected_output = expected_output
        self._state = state
        self._needs_revision = double_check
        self.instructions = expected_output if isinstance(expected_output, str) else None
        self.custom_schema = isinstance(expected_output, type)

    @property
    def input_schema(self) -> type[BaseModel]:
        return (
            self._expected_output
            if (
                self._expected_output is not None
                and isinstance(self._expected_output, type)
                and issubclass(self._expected_output, BaseModel)
            )
            else create_model(
                f"{self.name}Schema",
                response=(
                    str,
                    Field(description=self._expected_output or None),
                ),
            )
        )

    async def handler(self, obj: BaseModel, context: RunContext) -> StringToolOutput:
        if self.input_schema is self._expected_output:
            self._state.result = AssistantMessage(obj.model_dump_json())
        else:
            self._state.result = AssistantMessage(obj.response)  # type: ignore

        return StringToolOutput("Message has been sent")

    def check(self, state: ToolCallingAgentRunState) -> AgentAbilityState:
        return AgentAbilityState(allowed=True, forced=False, hidden=False, prevent_stop=False)


class ReasoningAbility(AgentAbility[BaseModel]):
    name = "Reasoning"
    description = 'Use this tool when you want to think through a problem, clarify your assumptions, or break down complex steps before acting or responding. This is your internal "scratchpad" — a place to reason out loud in natural language.'  # noqa: E501

    def __init__(self, force: bool) -> None:
        super().__init__()
        self.force = force

    @cached_property
    def input_schema(self) -> type[BaseModel]:
        class ReasoningSchema(BaseModel):
            thoughts: str = Field("Describe what you just saw and which tool will you use in the next step.")
            next_step: str = Field("Describe what will happen next.")

        return ReasoningSchema

    def check(self, states: ToolCallingAgentRunState) -> AgentAbilityState:
        last_step = states.steps[-1] if states.steps else None
        if last_step and last_step.tool and last_step.tool.name == self.name and not last_step.error:
            return AgentAbilityState(allowed=True, forced=False, hidden=False, prevent_stop=True)
        else:
            return AgentAbilityState(
                allowed=True, forced=self.force, hidden=False, prevent_stop=self.force and not last_step
            )

    async def handler(self, obj: BaseModel, context: RunContext) -> StringToolOutput:
        return StringToolOutput(
            "The observation seems reasonable. Remember that progress is made one step at a time. Stay determined and keep moving forward."  # noqa: E501
        )


class HandoffSchema(BaseModel):
    prompt: str = Field(description="Clearly defined task for the agent to work on based on his abilities.")


class HandoffAbility(AgentAbility[HandoffSchema]):
    """Delegates a task to an expert agent"""

    def __init__(self, target: "ToolCallingAgent", *, name: str | None = None, description: str | None = None) -> None:
        self._target = target

        self.name = name or target.meta.name
        self.description = description or target.meta.description
        # self.description += "(context must contain only verified information)"

        super().__init__()

    def check(self, states: ToolCallingAgentRunState) -> AgentAbilityState:
        return AgentAbilityState(allowed=True, forced=False, hidden=False, prevent_stop=False)

    async def handler(self, obj: HandoffSchema, context: RunContext) -> StringToolOutput:
        memory: BaseMemory = context.context["state"]["memory"]

        target = await self._target.clone()
        await target.memory.add_many(memory.messages[1:])
        response = await target.run(prompt=obj.prompt)
        return StringToolOutput(response.result.text)

    @cached_property
    def input_schema(self) -> type[HandoffSchema]:
        return HandoffSchema


class DynamicAbility(AgentAbility):
    """Ensures that a tool will be available only after previous one."""

    def __init__(
        self,
        name: str,
        description: str,
        handler: MaybeAsync[[Any, RunContext], Any],
        check: Callable[[Any], AgentAbilityState | bool],
        input_schema: type[BaseModel],
        emitter: Emitter | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._check = check
        self._handler = handler
        self._input_schema = input_schema
        self._emitter = emitter
        super().__init__()

    @property
    def input_schema(self) -> type[Any]:
        return self._input_schema

    @cached_property
    def emitter(self) -> Emitter:
        if self._emitter:
            print("yes")
            return self._emitter

        return super().emitter

    async def handler(self, input: Any, context: RunContext) -> Any:
        return await self._handler(input, context)

    def check(self, state: ToolCallingAgentRunState) -> AgentAbilityState:
        result = self._check(state)
        if isinstance(result, AgentAbilityState):
            return result
        else:
            return AgentAbilityState(allowed=bool(result), forced=False, hidden=False, prevent_stop=False)


def ability_from_tool(
    target: AnyTool, *, check: Callable[[Any], AgentAbilityState | bool] | None = None
) -> DynamicAbility:
    async def handler(input: Any, _: RunContext) -> Any:
        return await target.run(input)

    ability = DynamicAbility(
        name=target.name,
        description=target.description,
        handler=handler,
        check=(lambda state: True) if check is None else check,
        input_schema=target.input_schema,
    )
    return ability


class ToolAbility(Generic[TAbilityInput], AgentAbility[TAbilityInput]):
    def __init__(
        self,
        tool: Tool[TAbilityInput, Any, Any],
        *,
        force_at_step: int | None = None,
        only_before: list[str | AnyTool | AnyAbility] | None = None,
        only_after: list[str | AnyTool | AnyAbility] | None = None,
        force_after: list[str | AnyTool | AnyAbility] | None = None,
        max_invocations: int | None = None,
        required: bool = False,
        only_success_invocations: bool = True,
    ) -> None:
        self.tool = tool
        self.name = self.tool.name
        self.description = self.tool.description

        def extract_name(target: list[str | AnyTool | AnyAbility] | None) -> set[str]:
            return set[str](t if isinstance(t, str) else t.name for t in target or [])

        self._before = extract_name(only_before)
        self._after = extract_name(only_after)
        self._force_after = extract_name(force_after)
        self._max_invocations = max_invocations
        self._required = required
        self._force_at_step = force_at_step
        self._only_success_invocations = only_success_invocations

        self._check_invariant()

    def _check_invariant(self) -> None:
        if self.tool.name != self.name:
            raise ValueError(f"Tool name '{self.tool.name}' does not match ability name '{self.name}'")

        if self._required and self._max_invocations and self._max_invocations < 1:
            raise ValueError("Required tool must have at least one invocation!")

        if self.name in self._before:
            raise ValueError(f"Referencing self in 'before' is not allowed: {self.name}!")

        if self.name in self._force_after:
            raise ValueError(f"Referencing self in 'force_after' is not allowed: {self.name}!")

        before_after_force_req = self._before & self._after
        if before_after_force_req:
            raise ValueError(f"Tool specified as 'before' and 'after' at the same time: {before_after_force_req}!")

        before_after_force_req = self._before & self._force_after
        if before_after_force_req:
            raise ValueError(
                f"Tool specified as 'before' and 'force_after' at the same time: {before_after_force_req}!"
            )

        if (self._force_at_step or 0) < 0:
            raise ValueError("The 'force_at_step' argument must be non negative!")

    def verify(self, *, tools: list[AnyTool], abilities: list[AnyAbility]) -> None:
        existing_names = set([t.name for t in tools] + [a.name for a in abilities])

        def check(attr_name: str, target: set[str]) -> None:
            diff = target - existing_names
            if diff:
                raise ValueError(
                    f"Following tools ({diff}) are specified in '{attr_name}' but not found for the agent instance."
                )

        check("before", self._before)
        check("after", self._after)
        check("force_after", self._force_after)

    def reset(self) -> Self:
        self._before.clear()
        self._after.clear()
        self._force_after.clear()
        return self

    @property
    def input_schema(self) -> type[TAbilityInput]:
        return self.tool.input_schema

    async def handler(self, input: TAbilityInput, context: RunContext) -> Any:
        return await self.tool.run(input, ToolRunOptions(signal=context.signal))  # TODO: double check

    def check(self, state: ToolCallingAgentRunState) -> AgentAbilityState:
        steps = (
            [step for step in state.steps if not step.error] if self._only_success_invocations else list(state.steps)
        )

        def was_called_check(target: str) -> bool:
            return any(step.tool and step.tool.name == target for step in steps)

        def resolve(allowed: bool) -> AgentAbilityState:
            if not allowed and self._force_at_step == len(steps):
                raise ValueError(
                    f"Ability '{self.name}' cannot be executed at step {self._force_at_step} "
                    f"because it has not met all requirements."
                )

            if allowed:
                last_step = steps[-1] if steps else None
                last_tool_name = last_step.tool.name if last_step and last_step.tool else ""
                forced = last_tool_name in self._force_after or self._force_at_step == len(steps)
            else:
                forced = False

            return AgentAbilityState(
                allowed=allowed,
                forced=forced,
                hidden=False,
                prevent_stop=(self._required and invocations == 0) or forced,
            )

        invocations = sum(1 if step.tool and step.tool.name == self.tool.name else 0 for step in steps)
        if invocations >= (self._max_invocations or math.inf):
            return resolve(False)

        for target in self._before:
            if was_called_check(target):
                return resolve(False)

        for target in self._after:
            if not was_called_check(target):
                return resolve(False)

        return resolve(True)


AgentAbility.register("reasoning", lambda: ReasoningAbility(force=True))
AgentAbility.register("reasoning_optional", lambda: ReasoningAbility(force=False))

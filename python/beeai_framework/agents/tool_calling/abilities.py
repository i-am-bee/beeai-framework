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

from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

from beeai_framework.agents.tool_calling.types import (
    AgentAbility,
    AgentAbilityState,
    ToolCallingAgentRunState,
)
from beeai_framework.backend import AssistantMessage
from beeai_framework.context import RunContext
from beeai_framework.memory import BaseMemory
from beeai_framework.tools import StringToolOutput

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


AgentAbility.register("reasoning", lambda: ReasoningAbility(force=True))
AgentAbility.register("reasoning_optional", lambda: ReasoningAbility(force=False))

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

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Annotated, Any, ClassVar, Generic

from pydantic import BaseModel, ConfigDict, Field, InstanceOf
from typing_extensions import TypeVar

from beeai_framework.agents.tool_calling.prompts import (
    ToolCallingAgentCycleDetectionPrompt,
    ToolCallingAgentCycleDetectionPromptInput,
    ToolCallingAgentSystemPrompt,
    ToolCallingAgentSystemPromptInput,
    ToolCallingAgentTaskPrompt,
    ToolCallingAgentTaskPromptInput,
    ToolCallingAgentToolErrorPrompt,
    ToolCallingAgentToolErrorPromptInput,
)
from beeai_framework.backend import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    ToolMessage,
)
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import BaseMemory
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import AnyTool, Tool, ToolError, ToolOutput
from beeai_framework.tools import tool as create_tool
from beeai_framework.utils.strings import to_safe_word


class ToolCallingAgentTemplates(BaseModel):
    system: InstanceOf[PromptTemplate[ToolCallingAgentSystemPromptInput]] = Field(
        default_factory=lambda: ToolCallingAgentSystemPrompt.fork(None),
    )
    task: InstanceOf[PromptTemplate[ToolCallingAgentTaskPromptInput]] = Field(
        default_factory=lambda: ToolCallingAgentTaskPrompt.fork(None),
    )
    tool_error: InstanceOf[PromptTemplate[ToolCallingAgentToolErrorPromptInput]] = Field(
        default_factory=lambda: ToolCallingAgentToolErrorPrompt.fork(None),
    )
    cycle_detection: InstanceOf[PromptTemplate[ToolCallingAgentCycleDetectionPromptInput]] = Field(
        default_factory=lambda: ToolCallingAgentCycleDetectionPrompt.fork(None),
    )


ToolCallingAgentTemplateFactory = Callable[[InstanceOf[PromptTemplate[Any]]], InstanceOf[PromptTemplate[Any]]]
ToolCallingAgentTemplatesKeys = Annotated[str, lambda v: v in ToolCallingAgentTemplates.model_fields]


class ToolCallingAgentRunState(BaseModel):
    result: InstanceOf[AssistantMessage] | None = None
    memory: InstanceOf[BaseMemory]
    iteration: int
    steps: list["ToolCallingAgentRunStateStep"] = []


class ToolCallingAgentRunOutput(BaseModel):
    result: InstanceOf[AssistantMessage]
    memory: InstanceOf[BaseMemory]
    state: ToolCallingAgentRunState


class ToolCallingAgentRunStateStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    iteration: int
    tool: InstanceOf[Tool[Any, Any, Any]] | None
    input: dict[str, Any]
    output: InstanceOf[ToolOutput]
    ability: InstanceOf["AgentAbility[BaseModel]"] | None
    error: InstanceOf[FrameworkError] | None
    # extra: dict[str, Any]  # TODO: stored outputs from Abilities


class AgentAbilityState(BaseModel):
    allowed: bool = Field(True, description="Can the agent use the tool?")
    prevent_stop: bool = Field(False, description="Prevent the agent from calling the final answer.")
    forced: bool = Field(False, description="Must the agent use the tool?")
    hidden: bool = Field(False, description="Completely omit the tool.")

    def __bool__(self) -> bool:
        return self.allowed


TAbilityInput = TypeVar("TAbilityInput", bound=BaseModel, default=BaseModel)

AgentAbilityFactory = Callable[[], "AgentAbility[Any]"]


class AgentAbility(ABC, Generic[TAbilityInput]):
    name: str
    description: str
    state: dict[str, Any]
    priority: int

    def __init__(self, *args: Any, priority: int | None = None, **kwargs: Any) -> None:
        self.priority = priority or 10

    @abstractmethod
    async def handler(self, input: TAbilityInput, context: RunContext) -> Any: ...

    @abstractmethod
    def check(self, state: ToolCallingAgentRunState) -> AgentAbilityState: ...

    @cached_property
    def emitter(self) -> Emitter:
        emitter = Emitter.root().child(namespace=["ability", to_safe_word(self.name)])
        return emitter

    @property
    @abstractmethod
    def input_schema(self) -> type[TAbilityInput]: ...

    _registered_classes: ClassVar[dict[str, AgentAbilityFactory]] = {}

    def verify(self, *, tools: list[AnyTool], abilities: list["AnyAbility"]) -> None:
        # TODO: make required to call?
        pass

    @staticmethod
    def register(name: str, factory: AgentAbilityFactory) -> None:  # TODO: support cloneable?
        if name in AgentAbility._registered_classes:
            raise ValueError(f"Ability with name '{name}' has been already registered!")

        AgentAbility._registered_classes[name] = factory

    @staticmethod
    def lookup(name: str) -> "AgentAbility[Any]":  # TODO: add support for lookup by a signature instead
        factory = AgentAbility._registered_classes.get(name)
        if factory is None:
            raise ValueError(f"Ability with name '{name}' has not been registered!")
        return factory()

    def can_use(self, state: ToolCallingAgentRunState) -> AgentAbilityState:  # TODO: rename?
        response = self.check(state)
        return response if isinstance(response, AgentAbilityState) else AgentAbilityState(allowed=response)


class DynamicAgentAbility(AgentAbility[TAbilityInput]):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: type[TAbilityInput],
        handler: Callable[[TAbilityInput], Any] | None,
        check: Callable[[ToolCallingAgentRunState], AgentAbilityState | bool] | None,
    ) -> None:
        super().__init__()
        self.name = name
        self.description = description
        self._input_schema = input_schema
        self._handler = handler
        self._check = check

    async def handler(self, input: TAbilityInput, context: RunContext) -> Any:
        if not self._handler:
            return None

        if inspect.iscoroutinefunction(self._handler):
            return await self._handler(input)
        else:
            return self._handler(input)

    def check(self, state: ToolCallingAgentRunState) -> AgentAbilityState:
        response = self._check(state) if self._check else True
        if isinstance(response, bool):
            return AgentAbilityState(allowed=response, forced=False, hidden=False, prevent_stop=False)
        else:
            return response

    @property
    def input_schema(self) -> type[TAbilityInput]:
        return self._input_schema


AnyAbility = AgentAbility[Any]


def agent_ability(fn: Callable[..., Any]) -> AgentAbility[Any]:
    tool = create_tool(fn)
    return DynamicAgentAbility(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        # output_key=to_safe_word(tool.name),
        handler=fn,
        check=None,
    )


class ToolInvocationResult(BaseModel):
    msg: InstanceOf[MessageToolCallContent]
    tool: InstanceOf[Tool[Any, Any, Any]] | None
    input: dict[str, Any]
    output: InstanceOf[ToolOutput]
    error: InstanceOf[ToolError] | None

    def as_step(self, state: ToolCallingAgentRunState, ability: AnyAbility | None) -> ToolCallingAgentRunStateStep:
        return ToolCallingAgentRunStateStep(
            iteration=state.iteration,
            tool=self.tool,
            input=self.input,
            output=self.output,
            ability=ability,
            error=self.error,
        )

    def as_message(self) -> ToolMessage:
        return ToolMessage(
            MessageToolResultContent(
                tool_name=self.tool.name if self.tool else self.msg.tool_name,
                tool_call_id=self.msg.id,
                result=self.output.get_text_content(),
            )
        )

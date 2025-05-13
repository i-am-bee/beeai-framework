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

import asyncio
import contextlib
import json
from asyncio import create_task
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from beeai_framework.agents.tool_calling.abilities import FinalAnswerAbility
from beeai_framework.agents.tool_calling.prompts import (
    ToolCallingAgentSystemPromptInput,
    ToolTemplateDefinition,
)
from beeai_framework.agents.tool_calling.types import (
    AgentAbility,
    AnyAbility,
    DynamicAgentAbility,
    ToolCallingAgentRunState,
    ToolInvocationResult,
)
from beeai_framework.backend import MessageToolCallContent, SystemMessage
from beeai_framework.backend.types import ChatModelToolChoice
from beeai_framework.context import RunContext
from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import StringToolOutput, ToolError
from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.tools.tool import tool as create_tool
from beeai_framework.utils.strings import to_json


def assert_tools_uniqueness(tools: list[AnyTool]) -> None:
    seen = set()
    for tool in tools:
        if tool.name in seen:
            raise ValueError(f"Duplicate tool name '{tool.name}'!")
        seen.add(tool.name)


async def _run_tool(
    tools: list[AnyTool],
    msg: MessageToolCallContent,
    context: dict[str, Any],
) -> ToolInvocationResult:
    result = ToolInvocationResult(
        msg=msg,
        tool=None,
        input=json.loads(msg.args),
        output=StringToolOutput(""),
        error=None,
    )

    try:
        result.tool = next((tool for tool in tools if tool.name == msg.tool_name), None)
        if not result.tool:
            raise ToolError(f"Tool '{msg.tool_name}' does not exist!")

        result.output = await result.tool.run(result.input).context({**context, "tool_call_msg": msg})
    except ToolError as e:
        result.error = e
        result.output = StringToolOutput(e.explain())

    return result


async def _run_tools(
    tools: list[AnyTool], msgs: list[MessageToolCallContent], context: dict[str, Any]
) -> list[ToolInvocationResult]:
    return await asyncio.gather(
        *(create_task(_run_tool(tools, msg=msg, context=context)) for msg in msgs),
        return_exceptions=False,
    )


RegistryInput = AnyAbility | AnyTool


class ToolCallingAgentRequest(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    allowed_tools: list[AnyTool]
    hidden_tools: list[AnyTool]
    regular_tools: list[AnyTool]
    abilities_tools: list[AnyTool]
    ability_by_tool: dict[str, AnyAbility]
    tool_choice: ChatModelToolChoice
    final_answer: FinalAnswerAbility
    can_stop: bool


class ToolCallingAgentAbilitiesContainer:
    def __init__(
        self, *, tools: Sequence[AnyTool], abilities: Sequence[AnyAbility], final_answer: FinalAnswerAbility
    ) -> None:
        self.tools: dict[str, AnyTool] = {}
        self.abilities: dict[str, AnyAbility] = {}
        self.tool_by_ability = dict[AnyAbility, AnyTool]()
        self.ability_by_tool = dict[AnyTool, AnyAbility]()
        self.final_answer = final_answer
        self.update(tools=tools, abilities=abilities)

    def update(self, tools: Sequence[AnyTool], abilities: Sequence[AnyAbility]) -> None:
        self.tools.clear()
        self.ability_by_tool.clear()
        for tool in tools:
            self._register_tool(tool)

        self.abilities.clear()
        self.tool_by_ability.clear()
        for ability in [*abilities, self.final_answer]:
            self._register_ability(ability)

        self._verify()

    def _verify(self) -> None:
        all_tools = list(self.tools.values())
        all_abilities = list(self.abilities.values())

        for ability in self.abilities.values():
            ability.verify(tools=all_tools, abilities=all_abilities)

    def _register_tool(self, tool: AnyTool) -> None:
        self.tools[tool.name] = tool

        ability = DynamicAgentAbility(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            handler=tool.run,
            check=lambda _: True,
        )
        # TODO: emitter?
        self.ability_by_tool[tool] = ability

    def _register_ability(self, ability: AnyAbility) -> None:
        self.abilities[ability.name] = ability

        @create_tool(
            name=ability.name,
            description=ability.description,
            input_schema=ability.input_schema,
            with_context=True,
            emitter=ability.emitter.child(namespace=["tool"]),
        )
        async def tool(context: RunContext, **kwargs: Any) -> Any:
            input = ability.input_schema.model_validate(kwargs)
            return await ability.handler(input, context=context)

        self.tool_by_ability[ability] = tool


def _prepare_request(
    registry: ToolCallingAgentAbilitiesContainer, state: ToolCallingAgentRunState, force_tool_call: bool
) -> ToolCallingAgentRequest:
    tool_choice: Literal["required"] | AnyTool = "required"
    abilities_tools: list[AnyTool] = []
    regular_tools: list[AnyTool] = list(registry.tools.values())
    ability_by_tool: dict[str, AgentAbility] = {}
    allowed_tools: list[AnyTool] = [*regular_tools]
    hidden_tools: list[AnyTool] = []

    prevent_stop: bool = False
    forced_ability: AgentAbility | None = None

    for ability in registry.abilities.values():
        status = ability.can_use(state=state)

        tool = registry.tool_by_ability[ability]
        if status.forced:
            # abilities_tools.clear()
            # ability_by_tool.clear()
            allowed_tools.clear()
            # TODO: we need to update the selection of tools instead of removing them

        if status.hidden:
            hidden_tools.append(tool)

        abilities_tools.append(tool)
        ability_by_tool[tool.name] = ability

        if status.prevent_stop:
            prevent_stop = True

        if not status.allowed:
            continue

        allowed_tools.append(tool)
        if status.forced and (forced_ability is None or ability.priority > forced_ability.priority):
            forced_ability = ability
            tool_choice = tool

    if prevent_stop:
        with contextlib.suppress(ValueError):
            allowed_tools.remove(registry.tool_by_ability[registry.final_answer])

    if not allowed_tools:
        raise FrameworkError("Unknown state. Tools shouldn't not be empty.")
    if len(allowed_tools) == 1:
        tool_choice = allowed_tools[0]

    assert_tools_uniqueness(allowed_tools)
    return ToolCallingAgentRequest(
        allowed_tools=allowed_tools,
        regular_tools=regular_tools,
        abilities_tools=abilities_tools,
        ability_by_tool=ability_by_tool,
        tool_choice=tool_choice if isinstance(tool_choice, Tool) or force_tool_call or prevent_stop else "auto",
        final_answer=registry.final_answer,
        hidden_tools=hidden_tools,
        can_stop=not prevent_stop,
    )


def _create_system_message(
    *,
    template: PromptTemplate[ToolCallingAgentSystemPromptInput],
    final_answer: FinalAnswerAbility,
    hidden_tools: Sequence[AnyTool],
    regular_tools: Sequence[AnyTool],
    ability_tools: Sequence[AnyTool],
) -> SystemMessage:
    return SystemMessage(
        template.render(
            tools=[ToolTemplateDefinition.from_tool(tool, enabled=tool not in hidden_tools) for tool in regular_tools],
            abilities=[
                ToolTemplateDefinition.from_tool(tool, enabled=tool not in hidden_tools) for tool in ability_tools
            ],
            final_answer_tool=final_answer.name,
            final_answer_schema=to_json(final_answer.input_schema.model_json_schema(), indent=2, sort_keys=False)
            if final_answer.custom_schema
            else None,  # TODO: needed?
            final_answer_instructions=final_answer.instructions,  # TODO: update template!
        )
    )

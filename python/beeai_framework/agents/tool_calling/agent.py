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
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

from beeai_framework.agents import AgentError, AgentExecutionConfig
from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.tool_calling.abilities import FinalAnswerAbility
from beeai_framework.agents.tool_calling.events import (
    ToolCallingAgentStartEvent,
    ToolCallingAgentSuccessEvent,
    tool_calling_agent_event_types,
)
from beeai_framework.agents.tool_calling.prompts import ToolCallingAgentTaskPromptInput, ToolCallingAgentToolDefinition
from beeai_framework.agents.tool_calling.types import (
    AgentAbility,
    ToolCallingAgentRunOutput,
    ToolCallingAgentRunState,
    ToolCallingAgentTemplateFactory,
    ToolCallingAgentTemplates,
    ToolCallingAgentTemplatesKeys,
)
from beeai_framework.agents.tool_calling.utils import _run_tools, assert_tools_uniqueness
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.template import PromptTemplate
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils.counter import RetryCounter

__all__ = ["ToolCallingAgent"]

from beeai_framework.utils.strings import to_json


class ToolCallingAgent(BaseAgent[ToolCallingAgentRunOutput]):
    def __init__(
        self,
        *,
        llm: ChatModel,
        memory: BaseMemory | None = None,
        tools: Sequence[AnyTool] | None = None,
        templates: dict[ToolCallingAgentTemplatesKeys, PromptTemplate[Any] | ToolCallingAgentTemplateFactory]
        | None = None,
        abilities: Sequence[AgentAbility | str] | None = None,
        save_intermediate_steps: bool = True,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._memory = memory or UnconstrainedMemory()
        self._tools = tools or []
        self._templates = self._generate_templates(templates)
        self._abilities = [AgentAbility.lookup(ab) if isinstance(ab, str) else ab for ab in (abilities or [])]
        self._save_intermediate_steps = save_intermediate_steps

    def run(
        self,
        prompt: str | None = None,
        *,
        context: str | None = None,
        expected_output: str | type[BaseModel] | None = None,
        execution: AgentExecutionConfig | None = None,
    ) -> Run[ToolCallingAgentRunOutput]:
        execution_config = execution or AgentExecutionConfig(
            max_retries_per_step=3,
            total_max_retries=20,
            max_iterations=20,
        )

        def prepare_request(
            state: ToolCallingAgentRunState,
            final_answer_ability: FinalAnswerAbility,
        ) -> tuple[
            list[AnyTool],
            list[AnyTool],
            list[AnyTool],
            dict[AnyTool, AgentAbility],
            Literal["required"] | AnyTool,
        ]:
            tool_choice: Literal["required"] | AnyTool = "required"
            abilities_tools: list[AnyTool] = []
            regular_tools: list[AnyTool] = list(self._tools)
            ability_by_tool: dict[AnyTool, AgentAbility] = {}
            allowed_tools: list[AnyTool] = [*regular_tools]

            for ability in [*self._abilities, final_answer_ability]:
                status = ability.can_use(state=state)
                if not status.allowed:
                    continue

                tool = ability.to_tool()
                if status.forced:
                    abilities_tools.clear()
                    ability_by_tool.clear()
                    allowed_tools.clear()
                    # we need to update the selection of tools instead of removing them!

                abilities_tools.append(tool)
                ability_by_tool[tool] = ability
                allowed_tools.append(tool)

                if status.forced and status.prevent_stop:
                    tool_choice = tool
                    break

            if not allowed_tools:
                raise FrameworkError("Unknown state. Tools shouldn't not be empty.")
            if len(allowed_tools) == 1:
                tool_choice = allowed_tools[0]

            assert_tools_uniqueness(allowed_tools)
            return allowed_tools, regular_tools, abilities_tools, ability_by_tool, tool_choice

        async def handler(run_context: RunContext) -> ToolCallingAgentRunOutput:
            state = ToolCallingAgentRunState(memory=UnconstrainedMemory(), steps=[], iteration=0, result=None)
            await state.memory.add_many(self.memory.messages)

            user_message: UserMessage | None = None
            if prompt:
                task_input = ToolCallingAgentTaskPromptInput(
                    prompt=prompt,
                    context=context,
                    expected_output=expected_output if isinstance(expected_output, str) else None,
                )
                user_message = UserMessage(self._templates.task.render(task_input))
                await state.memory.add(user_message)

            global_retries_counter = RetryCounter(
                error_type=AgentError, max_retries=execution_config.total_max_retries or 1
            )

            final_answer_ability = FinalAnswerAbility(state=state, expected_output=expected_output)

            while state.result is None:
                state.iteration += 1

                if execution_config.max_iterations and state.iteration > execution_config.max_iterations:
                    raise AgentError(f"Agent was not able to resolve the task in {state.iteration} iterations.")

                allowed_tools, regular_tools, abilities_tools, ability_by_tool, tool_choice = prepare_request(
                    state, final_answer_ability
                )

                await run_context.emitter.emit(
                    "start",
                    ToolCallingAgentStartEvent(state=state),
                )
                system_message = SystemMessage(
                    self._templates.system.render(
                        tools=[
                            ToolCallingAgentToolDefinition.from_tool(tool, enabled=tool in allowed_tools)
                            for tool in regular_tools
                        ],
                        abilities=[
                            ToolCallingAgentToolDefinition.from_tool(tool, enabled=tool in allowed_tools)
                            for tool in abilities_tools
                        ],
                        final_answer_tool=final_answer_ability.name,
                        final_answer_schema=to_json(
                            final_answer_ability.input_schema.model_json_schema(), indent=2, sort_keys=False
                        )
                        if isinstance(expected_output, type)
                        else expected_output,
                    )
                )
                response = await self._llm.create(
                    messages=[
                        system_message,
                        *state.memory.messages,
                    ],
                    tools=allowed_tools,
                    tool_choice=tool_choice,
                    stream=False,
                )
                await state.memory.add_many(response.messages)

                tool_call_messages = response.get_tool_calls()
                for tool_call in await _run_tools(
                    allowed_tools, tool_call_messages, context={"state": state.model_dump()}
                ):
                    state.steps.append(tool_call.to_step(state, ability_by_tool))
                    await state.memory.add(tool_call.to_message())
                    if tool_call.error:
                        global_retries_counter.use(tool_call.error)

                # handle empty messages for some models
                text_messages = response.get_text_messages()
                if not tool_call_messages and not text_messages:
                    await state.memory.add(AssistantMessage("\n", {"tempMessage": True}))
                else:
                    await state.memory.delete_many(
                        [msg for msg in state.memory.messages if msg.meta.get("tempMessage", False)]
                    )

                await run_context.emitter.emit(
                    "success",
                    ToolCallingAgentSuccessEvent(state=state),
                )

            if self._save_intermediate_steps:
                self.memory.reset()
                await self.memory.add_many(state.memory.messages)
            else:
                if user_message is not None:
                    await self.memory.add(user_message)
                await self.memory.add_many(state.memory.messages[-2:])

            assert state.result is not None
            return ToolCallingAgentRunOutput(result=state.result, memory=state.memory, state=state)

        return self._to_run(handler, signal=None, run_params={"prompt": prompt, "execution": execution})

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "tool_calling"], creator=self, events=tool_calling_agent_event_types
        )

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._memory = memory

    @staticmethod
    def _generate_templates(
        overrides: dict[ToolCallingAgentTemplatesKeys, PromptTemplate[Any] | ToolCallingAgentTemplateFactory]
        | None = None,
    ) -> ToolCallingAgentTemplates:
        templates = ToolCallingAgentTemplates()
        if overrides is None:
            return templates

        for name, _info in ToolCallingAgentTemplates.model_fields.items():
            override: PromptTemplate[Any] | ToolCallingAgentTemplateFactory | None = overrides.get(name)
            if override is None:
                continue
            elif isinstance(override, PromptTemplate):
                setattr(templates, name, override)
            else:
                setattr(templates, name, override(getattr(templates, name)))
        return templates

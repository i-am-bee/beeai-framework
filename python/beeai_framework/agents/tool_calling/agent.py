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

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, create_model

from beeai_framework.agents import AgentError, AgentExecutionConfig
from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.tool_calling.events import (
    ToolCallingAgentStartEvent,
    ToolCallingAgentSuccessEvent,
    tool_calling_agent_event_types,
)
from beeai_framework.agents.tool_calling.prompts import ToolCallingAgentTaskPromptInput
from beeai_framework.agents.tool_calling.types import (
    ToolCallingAgentRunOutput,
    ToolCallingAgentRunState,
    ToolCallingAgentTemplateFactory,
    ToolCallingAgentTemplates,
    ToolCallingAgentTemplatesKeys,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolResultContent,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.template import PromptTemplate
from beeai_framework.tools.errors import ToolError
from beeai_framework.tools.tool import AnyTool
from beeai_framework.tools.tool import tool as create_tool
from beeai_framework.tools.types import StringToolOutput
from beeai_framework.utils.counter import RetryCounter
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
        save_intermediate_steps: bool = True,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._memory = memory or UnconstrainedMemory()
        self._tools = tools or []
        self._templates = self._generate_templates(templates)
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
            max_iterations=10,
        )

        async def handler(run_context: RunContext) -> ToolCallingAgentRunOutput:
            state = ToolCallingAgentRunState(memory=UnconstrainedMemory(), result=None, iteration=0)
            await state.memory.add(SystemMessage(self._templates.system.render()))
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

            final_answer_schema_cls: type[BaseModel] = (
                expected_output
                if (
                    expected_output is not None
                    and isinstance(expected_output, type)
                    and issubclass(expected_output, BaseModel)
                )
                else create_model(
                    "FinalAnswer",
                    response=(
                        str,
                        Field(description=expected_output or None),
                    ),
                )
            )

            @create_tool(
                name="final_answer",
                description="Sends the final answer to the user",
                input_schema=final_answer_schema_cls,
            )
            def final_answer_tool(**kwargs: Any) -> StringToolOutput:
                if final_answer_schema_cls is expected_output:
                    dump = final_answer_schema_cls.model_validate(kwargs)
                    state.result = AssistantMessage(to_json(dump.model_dump()))
                else:
                    state.result = AssistantMessage(kwargs["response"])

                return StringToolOutput("Message has been sent")

            tools = [*self._tools, final_answer_tool]

            while state.result is None:
                state.iteration += 1

                if execution_config.max_iterations and state.iteration > execution_config.max_iterations:
                    raise AgentError(f"Agent was not able to resolve the task in {state.iteration} iterations.")

                await run_context.emitter.emit(
                    "start",
                    ToolCallingAgentStartEvent(state=state),
                )
                response = await self._llm.create(
                    messages=state.memory.messages,
                    tools=tools,
                    tool_choice="required" if len(tools) > 1 else tools[0],
                    stream=False,
                )
                await state.memory.add_many(response.messages)

                tool_call_messages = response.get_tool_calls()
                for tool_call in tool_call_messages:
                    try:
                        tool = next((tool for tool in tools if tool.name == tool_call.tool_name), None)
                        if not tool:
                            raise ToolError(f"Tool '{tool_call.tool_name}' does not exist!")

                        tool_input = json.loads(tool_call.args)
                        tool_response = await tool.run(tool_input).context(
                            {"state": state.model_dump(), "tool_call_msg": tool_call}
                        )
                        await state.memory.add(
                            ToolMessage(
                                MessageToolResultContent(
                                    result=tool_response.get_text_content(),
                                    tool_name=tool_call.tool_name,
                                    tool_call_id=tool_call.id,
                                )
                            )
                        )
                    except ToolError as e:
                        global_retries_counter.use(e)
                        await state.memory.add(
                            ToolMessage(
                                MessageToolResultContent(
                                    result=self._templates.tool_error.render({"reason": e.explain()}),
                                    tool_name=tool_call.tool_name,
                                    tool_call_id=tool_call.id,
                                )
                            )
                        )

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

            assert state.result is not None
            if self._save_intermediate_steps:
                self.memory.reset()
                await self.memory.add_many(state.memory.messages[1:])
            else:
                if user_message is not None:
                    await self.memory.add(user_message)
                await self.memory.add_many(state.memory.messages[-2:])
            return ToolCallingAgentRunOutput(result=state.result, memory=state.memory)

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

    async def clone(self) -> "ToolCallingAgent":
        cloned = ToolCallingAgent(
            llm=await self._llm.clone(),
            memory=await self._memory.clone(),
            tools=[await tool.clone() for tool in self._tools],
            templates=self._templates.model_dump(),
        )
        cloned.emitter = await self.emitter.clone()
        return cloned

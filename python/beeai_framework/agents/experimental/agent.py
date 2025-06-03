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

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from beeai_framework.agents import AgentError, AgentExecutionConfig, AgentMeta
from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.experimental._utils import _create_system_message
from beeai_framework.agents.experimental.events import (
    RequirementAgentStartEvent,
    RequirementAgentSuccessEvent,
    requirement_agent_event_types,
)
from beeai_framework.agents.experimental.prompts import (
    RequirementAgentCycleDetectionPromptInput,
    RequirementAgentTaskPromptInput,
)
from beeai_framework.agents.experimental.requirements.requirement import Requirement
from beeai_framework.agents.experimental.types import (
    RequirementAgentRunOutput,
    RequirementAgentRunState,
    RequirementAgentRunStateStep,
    RequirementAgentTemplateFactory,
    RequirementAgentTemplates,
    RequirementAgentTemplatesKeys,
)
from beeai_framework.agents.experimental.utils._llm import RequirementsReasoner
from beeai_framework.agents.experimental.utils._tool import FinalAnswerTool, _run_tools
from beeai_framework.agents.tool_calling.utils import ToolCallChecker, ToolCallCheckerConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    ToolMessage,
    UserMessage,
)
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.memory.utils import extract_last_tool_call_pair
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import AnyTool
from beeai_framework.utils.counter import RetryCounter
from beeai_framework.utils.dicts import exclude_none
from beeai_framework.utils.lists import cast_list
from beeai_framework.utils.models import update_model
from beeai_framework.utils.strings import find_first_pair, generate_random_string, to_json

RequirementAgentRequirement = Requirement[RequirementAgentRunState]


class RequirementAgent(BaseAgent[RequirementAgentRunOutput]):
    def __init__(
        self,
        *,
        llm: ChatModel,
        memory: BaseMemory | None = None,
        tools: Sequence[AnyTool] | None = None,
        requirements: Sequence[RequirementAgentRequirement] | None = None,
        name: str | None = None,
        description: str | None = None,
        role: str | None = None,
        instructions: str | list[str] | None = None,
        notes: str | list[str] | None = None,
        tool_call_checker: ToolCallCheckerConfig | bool = True,
        final_answer_as_tool: bool = True,
        save_intermediate_steps: bool = True,
        templates: dict[RequirementAgentTemplatesKeys, PromptTemplate[Any] | RequirementAgentTemplateFactory]
        | RequirementAgentTemplates
        | None = None,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._memory = memory or UnconstrainedMemory()
        self._templates = self._generate_templates(templates)
        self._save_intermediate_steps = save_intermediate_steps
        self._tool_call_checker = tool_call_checker
        self._final_answer_as_tool = final_answer_as_tool
        if role or instructions or notes:
            self._templates.system.update(
                defaults=exclude_none(
                    {
                        "role": role,
                        "instructions": "\n -".join(cast_list(instructions)) if instructions else None,
                        "notes": "\n -".join(cast_list(notes)) if notes else None,
                    }
                )
            )
        self._tools = list(tools or [])
        self._requirements = list(requirements or [])
        self._meta = AgentMeta(name=name or "", description=description or "", tools=self._tools)

    def run(
        self,
        prompt: str | None = None,
        *,
        context: str | None = None,
        expected_output: str | type[BaseModel] | None = None,
        execution: AgentExecutionConfig | None = None,
    ) -> Run[RequirementAgentRunOutput]:
        run_config = execution or AgentExecutionConfig(
            max_retries_per_step=3,
            total_max_retries=20,
            max_iterations=10,
        )

        async def init_state() -> tuple[RequirementAgentRunState, UserMessage | None]:
            state = RequirementAgentRunState(memory=UnconstrainedMemory(), steps=[], iteration=0, result=None)
            await state.memory.add_many(self.memory.messages)

            user_message: UserMessage | None = None
            if prompt:
                task_input = RequirementAgentTaskPromptInput(
                    prompt=prompt,
                    context=context,
                    expected_output=expected_output if isinstance(expected_output, str) else None,  # TODO: validate
                )
                user_message = UserMessage(self._templates.task.render(task_input))
                await state.memory.add(user_message)

            return state, user_message

        async def handler(run_context: RunContext) -> RequirementAgentRunOutput:
            state, user_message = await init_state()

            reasoner = RequirementsReasoner(
                tools=self._tools,
                requirements=self._requirements,
                final_answer=FinalAnswerTool(expected_output, state=state),
                context=run_context,
            )
            tool_call_cycle_checker = self._create_tool_call_checker()
            tool_call_retry_counter = RetryCounter(error_type=AgentError, max_retries=run_config.total_max_retries or 1)
            force_final_answer_as_tool = self._final_answer_as_tool

            while state.result is None:
                state.iteration += 1

                if run_config.max_iterations and state.iteration > run_config.max_iterations:
                    raise AgentError(f"Agent was not able to resolve the task in {state.iteration} iterations.")

                request = await reasoner.create_request(state, force_tool_call=force_final_answer_as_tool)

                await run_context.emitter.emit(
                    "start",
                    RequirementAgentStartEvent(state=state, request=request),
                )

                response = await self._llm.create(
                    messages=[
                        _create_system_message(
                            template=self._templates.system,
                            request=request,
                        ),
                        *state.memory.messages,
                    ],
                    tools=request.allowed_tools,
                    tool_choice=request.tool_choice,
                    stream=False,
                )
                await state.memory.add_many(response.messages)

                text_messages = response.get_text_messages()
                tool_call_messages = response.get_tool_calls()

                if not tool_call_messages and text_messages and request.can_stop:
                    await state.memory.delete_many(response.messages)

                    full_text = "".join(msg.text for msg in text_messages)
                    json_object_pair = find_first_pair(full_text, ("{", "}"))
                    final_answer_input = parse_broken_json(json_object_pair.outer) if json_object_pair else None
                    if not final_answer_input and not request.final_answer.custom_schema:
                        final_answer_input = {"response": full_text}

                    if not final_answer_input:
                        reasoner.update(requirements=[])
                        force_final_answer_as_tool = True
                        continue

                    tool_call_message = MessageToolCallContent(
                        type="tool-call",
                        id=f"call_{generate_random_string(8).lower()}",
                        tool_name=reasoner.final_answer.name,
                        args=to_json(final_answer_input, sort_keys=False),
                    )
                    tool_call_messages.append(tool_call_message)
                    await state.memory.add(AssistantMessage(tool_call_message))

                cycle_found = False
                for tool_call_msg in tool_call_messages:
                    tool_call_cycle_checker.register(tool_call_msg)
                    if cycle_found := tool_call_cycle_checker.cycle_found:
                        await state.memory.delete_many(response.messages)
                        await state.memory.add(
                            UserMessage(
                                self._templates.cycle_detection.render(
                                    RequirementAgentCycleDetectionPromptInput(
                                        tool_args=tool_call_msg.args,
                                        tool_name=tool_call_msg.tool_name,
                                        final_answer_name=request.final_answer.name,
                                    )
                                )
                            )
                        )
                        tool_call_cycle_checker.reset()
                        break

                if not cycle_found:
                    for tool_call in await _run_tools(
                        tools=request.allowed_tools,
                        messages=tool_call_messages,
                        context={"state": state.model_dump()},
                    ):
                        state.steps.append(
                            RequirementAgentRunStateStep(
                                iteration=state.iteration,
                                input=tool_call.input,
                                output=tool_call.output,
                                tool=tool_call.tool,
                                error=tool_call.error,
                            )
                        )
                        await state.memory.add(
                            ToolMessage(
                                MessageToolResultContent(
                                    tool_name=tool_call.tool.name if tool_call.tool else tool_call.msg.tool_name,
                                    tool_call_id=tool_call.msg.id,
                                    result=tool_call.output.get_text_content()
                                    if not tool_call.output.is_empty()
                                    else self._templates.tool_no_result.render(tool_call=tool_call),
                                )
                            )
                        )
                        if tool_call.error:
                            tool_call_retry_counter.use(tool_call.error)

                # handle empty responses for some models
                if not tool_call_messages and not text_messages:
                    await state.memory.add(AssistantMessage("\n", {"tempMessage": True}))
                else:
                    await state.memory.delete_many(
                        [msg for msg in state.memory.messages if msg.meta.get("tempMessage", False)]
                    )

                await run_context.emitter.emit(
                    "success",
                    RequirementAgentSuccessEvent(state=state, response=response),
                )

            if self._save_intermediate_steps:
                self.memory.reset()
                await self.memory.add_many(state.memory.messages)
            else:
                if user_message is not None:
                    await self.memory.add(user_message)

                await self.memory.add_many(extract_last_tool_call_pair(state.memory) or [])

            return RequirementAgentRunOutput(result=state.result, memory=state.memory, state=state)

        return self._to_run(
            handler,
            signal=None,
            run_params={
                "prompt": prompt,
                "context": context,
                "expected_output": expected_output,
                "execution": execution,
            },
        )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "requirement"], creator=self, events=requirement_agent_event_types
        )

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._memory = memory

    @staticmethod
    def _generate_templates(
        overrides: dict[RequirementAgentTemplatesKeys, PromptTemplate[Any] | RequirementAgentTemplateFactory]
        | RequirementAgentTemplates
        | None = None,
    ) -> RequirementAgentTemplates:
        if isinstance(overrides, RequirementAgentTemplates):
            return overrides

        templates = RequirementAgentTemplates()
        if overrides is None:
            return templates

        for name, _info in RequirementAgentTemplates.model_fields.items():
            override: PromptTemplate[Any] | RequirementAgentTemplateFactory | None = overrides.get(name)
            if override is None:
                continue
            elif isinstance(override, PromptTemplate):
                setattr(templates, name, override)
            else:
                setattr(templates, name, override(getattr(templates, name)))
        return templates

    async def clone(self) -> "RequirementAgent":
        cloned = RequirementAgent(
            llm=await self._llm.clone(),
            memory=await self._memory.clone(),
            tools=self._tools.copy(),
            requirements=self._requirements.copy(),
            templates=self._templates.model_dump(),
            tool_call_checker=self._tool_call_checker,
            save_intermediate_steps=self._save_intermediate_steps,
            final_answer_as_tool=self._final_answer_as_tool,
            name=self._meta.name,
            description=self._meta.description,
        )
        cloned.emitter = await self.emitter.clone()
        return cloned

    @property
    def meta(self) -> AgentMeta:
        parent = super().meta

        return AgentMeta(
            name=self._meta.name or parent.name,
            description=self._meta.description or parent.description,
            extra_description=self._meta.extra_description or parent.extra_description,
            tools=list(self._tools),
        )

    def _create_tool_call_checker(self) -> ToolCallChecker:
        config = ToolCallCheckerConfig()
        update_model(config, sources=[self._tool_call_checker])

        instance = ToolCallChecker(config)
        instance.enabled = self._tool_call_checker is not False
        return instance

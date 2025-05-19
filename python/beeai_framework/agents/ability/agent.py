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
from beeai_framework.agents.ability._utils import _create_system_message
from beeai_framework.agents.ability.abilities.ability import Ability
from beeai_framework.agents.ability.abilities.final_answer import FinalAnswerAbility
from beeai_framework.agents.ability.abilities.tool import ToolAbility
from beeai_framework.agents.ability.events import (
    AbilityAgentStartEvent,
    AbilityAgentSuccessEvent,
    ability_agent_event_types,
)
from beeai_framework.agents.ability.prompts import AbilityAgentCycleDetectionPromptInput, AbilityAgentTaskPromptInput
from beeai_framework.agents.ability.types import (
    AbilityAgentRunOutput,
    AbilityAgentRunState,
    AbilityAgentRunStateStep,
    AbilityAgentTemplateFactory,
    AbilityAgentTemplates,
    AbilityAgentTemplatesKeys,
)
from beeai_framework.agents.ability.utils._llm import AbilityModel
from beeai_framework.agents.ability.utils._tool import _run_abilities
from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.tool_calling.utils import ToolCallChecker, ToolCallCheckerConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    UserMessage,
)
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.context import Run, RunContext, RunMiddleware
from beeai_framework.emitter import Emitter
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.memory.utils import extract_last_tool_call_pair
from beeai_framework.template import PromptTemplate
from beeai_framework.utils.counter import RetryCounter
from beeai_framework.utils.dicts import exclude_none
from beeai_framework.utils.models import update_model
from beeai_framework.utils.strings import find_first_pair, generate_random_string, to_json

AbilityAgentAbility = Ability[Any, AbilityAgentRunState, Any]


class AbilityAgent(BaseAgent[AbilityAgentRunOutput]):
    def __init__(
        self,
        *,
        llm: ChatModel,
        memory: BaseMemory | None = None,
        templates: dict[AbilityAgentTemplatesKeys, PromptTemplate[Any] | AbilityAgentTemplateFactory] | None = None,
        save_intermediate_steps: bool = True,
        abilities: Sequence[AbilityAgentAbility | str] | None = None,
        name: str | None = None,
        description: str | None = None,
        role: str | None = None,
        instructions: str | None = None,
        tool_call_checker: ToolCallCheckerConfig | bool = True,
        final_answer_as_tool: bool = True,
        meta: AgentMeta | None = None,  # TODO: probably remove
    ) -> None:
        super().__init__()
        self._llm = llm
        self._memory = memory or UnconstrainedMemory()
        self._templates = self._generate_templates(templates)
        self._save_intermediate_steps = save_intermediate_steps
        self._tool_call_checker = tool_call_checker
        self._final_answer_as_tool = final_answer_as_tool
        if role or instructions:
            self._templates.system.update(
                defaults=exclude_none(
                    {
                        "role": role,
                        "instructions": instructions,
                    }
                )
            )
        self._abilities = [Ability.lookup(ab) if isinstance(ab, str) else ab for ab in (abilities or [])]
        self._meta = AgentMeta(name=name or "", description=description or instructions or "", tools=[])
        self.middlewares: list[RunMiddleware] = []
        if meta:
            update_model(self._meta, sources=[meta])

    def run(
        self,
        prompt: str | None = None,
        *,
        context: str | None = None,
        expected_output: str | type[BaseModel] | None = None,
        execution: AgentExecutionConfig | None = None,
    ) -> Run[AbilityAgentRunOutput]:
        run_config = execution or AgentExecutionConfig(
            max_retries_per_step=3,
            total_max_retries=20,
            max_iterations=10,
        )

        async def init_state() -> tuple[AbilityAgentRunState, UserMessage | None]:
            state = AbilityAgentRunState(memory=UnconstrainedMemory(), steps=[], iteration=0, result=None)
            await state.memory.add_many(self.memory.messages)

            user_message: UserMessage | None = None
            if prompt:
                task_input = AbilityAgentTaskPromptInput(
                    prompt=prompt,
                    context=context,
                    expected_output=expected_output if isinstance(expected_output, str) else None,  # TODO: validate
                )
                user_message = UserMessage(self._templates.task.render(task_input))
                await state.memory.add(user_message)

            return state, user_message

        async def handler(run_context: RunContext) -> AbilityAgentRunOutput:
            for middleware in self.middlewares:
                middleware.bind(run_context)

            state, user_message = await init_state()
            ability_model = AbilityModel(
                abilities=self._abilities,
                final_answer=FinalAnswerAbility(state=state, expected_output=expected_output, double_check=False),
            )
            tool_call_cycle_checker = self._create_tool_call_checker()
            tool_call_retry_counter = RetryCounter(error_type=AgentError, max_retries=run_config.total_max_retries or 1)
            force_final_answer_as_tool = self._final_answer_as_tool

            while state.result is None:
                state.iteration += 1

                if run_config.max_iterations and state.iteration > run_config.max_iterations:
                    raise AgentError(f"Agent was not able to resolve the task in {state.iteration} iterations.")

                request = ability_model.create_request(state, force_tool_call=force_final_answer_as_tool)

                await run_context.emitter.emit(
                    "start",
                    AbilityAgentStartEvent(state=state, request=request),
                )

                response = await self._llm.create(
                    messages=[
                        # TODO: pass abilities instead!
                        _create_system_message(
                            template=self._templates.system,
                            request=request,
                        ),
                        *state.memory.messages,
                    ],
                    tools=[entry.tool for entry in request.allowed],
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
                    if not final_answer_input and not ability_model.final_answer.ability.custom_schema:
                        final_answer_input = {"response": full_text}

                    if not final_answer_input:
                        ability_model.update(abilities=[])
                        force_final_answer_as_tool = True
                        continue

                    tool_call_message = MessageToolCallContent(
                        type="tool-call",
                        id=f"call_{generate_random_string(8).lower()}",
                        tool_name=ability_model.final_answer.name,
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
                                    AbilityAgentCycleDetectionPromptInput(
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
                    # task by ability
                    for tool_call in await _run_abilities(
                        abilities=request.allowed, messages=tool_call_messages, context={"state": state.model_dump()}
                    ):
                        state.steps.append(
                            AbilityAgentRunStateStep(
                                iteration=state.iteration,
                                input=tool_call.input,
                                output=tool_call.output,
                                ability=tool_call.ability.ability if tool_call.ability else None,  # TODO: refactor?
                                error=tool_call.error,
                            )
                        )
                        await state.memory.add(tool_call.as_message())
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
                    AbilityAgentSuccessEvent(state=state),
                )

            if self._save_intermediate_steps:
                self.memory.reset()
                await self.memory.add_many(state.memory.messages)
            else:
                if user_message is not None:
                    await self.memory.add(user_message)

                await self.memory.add_many(extract_last_tool_call_pair(state.memory) or [])

            return AbilityAgentRunOutput(result=state.result, memory=state.memory, state=state)

        return self._to_run(handler, signal=None, run_params={"prompt": prompt, "execution": execution})

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["agent", "ability"], creator=self, events=ability_agent_event_types)

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._memory = memory

    @staticmethod
    def _generate_templates(
        overrides: dict[AbilityAgentTemplatesKeys, PromptTemplate[Any] | AbilityAgentTemplateFactory] | None = None,
    ) -> AbilityAgentTemplates:
        templates = AbilityAgentTemplates()
        if overrides is None:
            return templates

        for name, _info in AbilityAgentTemplates.model_fields.items():
            override: PromptTemplate[Any] | AbilityAgentTemplateFactory | None = overrides.get(name)
            if override is None:
                continue
            elif isinstance(override, PromptTemplate):
                setattr(templates, name, override)
            else:
                setattr(templates, name, override(getattr(templates, name)))
        return templates

    async def clone(self) -> "AbilityAgent":
        cloned = AbilityAgent(
            llm=await self._llm.clone(),
            memory=await self._memory.clone(),
            abilities=[await ability.clone() for ability in self._abilities],
            templates=self._templates.model_dump(),
            tool_call_checker=self._tool_call_checker,
            save_intermediate_steps=self._save_intermediate_steps,
            meta=self._meta,
            final_answer_as_tool=self._final_answer_as_tool,
            # TODO: middlewares
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
            tools=[
                ability.tool for ability in self._abilities if isinstance(ability, ToolAbility)
            ],  # TODO: only filter ToolAbility or convert all?
        )

    def _create_tool_call_checker(self) -> ToolCallChecker:
        config = ToolCallCheckerConfig()
        update_model(config, sources=[self._tool_call_checker])

        instance = ToolCallChecker(config)
        instance.enabled = self._tool_call_checker is not False
        return instance

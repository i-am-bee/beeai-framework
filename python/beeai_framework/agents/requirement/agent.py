# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from typing_extensions import Unpack

from beeai_framework.agents import AgentExecutionConfig, AgentMeta, AgentOptions, BaseAgent
from beeai_framework.agents.requirement._runner import RequirementAgentRunner
from beeai_framework.agents.requirement.events import (
    requirement_agent_event_types,
)
from beeai_framework.agents.requirement.prompts import (
    RequirementAgentTaskPromptInput,
)
from beeai_framework.agents.requirement.requirements.requirement import Requirement
from beeai_framework.agents.requirement.types import (
    RequirementAgentOutput,
    RequirementAgentRunState,
    RequirementAgentTemplateFactory,
    RequirementAgentTemplates,
    RequirementAgentTemplatesKeys,
)
from beeai_framework.agents.tool_calling.utils import ToolCallChecker, ToolCallCheckerConfig
from beeai_framework.backend import AnyMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import (
    MessageTextContent,
    UserMessage,
)
from beeai_framework.context import RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.memory.utils import extract_last_tool_call_pair
from beeai_framework.runnable import runnable_entry
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import AnyTool
from beeai_framework.utils.dicts import exclude_none
from beeai_framework.utils.lists import cast_list
from beeai_framework.utils.models import update_model

RequirementAgentRequirement = Requirement[RequirementAgentRunState]


class RequirementAgent(BaseAgent[RequirementAgentOutput]):
    """
    The RequirementAgent is a declarative AI agent implementation that provides predictable,
    controlled execution behavior across different language models through rule-based constraints.
    Language models vary significantly in their reasoning capabilities and tool-calling sophistication, but
    RequirementAgent normalizes these differences by enforcing consistent execution patterns
    regardless of the underlying model's strengths or weaknesses.
    Rules can be configured as strict or flexible as necessary, adapting to task requirements while ensuring consistent
    execution regardless of the underlying model's reasoning or tool-calling capabilities.
    """

    def __init__(
        self,
        *,
        llm: ChatModel | str,
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
        middlewares: list[RunMiddlewareType] | None = None,
    ) -> None:
        """
        Initializes an instance of the RequirementAgent class.

        Args:
            llm:
                The language model to be used for chat functionality. Can be provided as
                an instance of ChatModel or as a string representing the model name.

            tools:
                A sequence of tools that the agent can use during the execution. Default is an empty list.

            requirements:
                A sequence of requirements that constrain the agent's behavior.

            memory:
                The memory instance to store conversation history or state. If none is
                provided, a default UnconstrainedMemory instance will be used.

            name:
                A name of the agent which should emphasize its purpose.
                This property is used in multi-agent components like HandoffTool or when exposing the agent as a server.

            description:
                A brief description of the agent abilities.
                This property is used in multi-agent components like HandoffTool or when exposing the agent as a server.

            role:
                Role for the agent. Will be part of the system prompt.

            instructions:
                Instructions for the agents. Will be part of the system prompt. Can be a single string or a list of
                strings. If a list is provided, it will be formatted as a single newline-separated string.

            save_intermediate_steps:
                Determines whether intermediate steps during execution should be preserved between individual turns.
                If enabled (default), the agent can reuse existing tool results and might provide a better result
                  but consumes more tokens.

            middlewares:
                A list of middlewares to be applied for an upcoming execution.
                Useful for logging and altering the behavior.

            templates:
                Templates define prompts that the model will work with. Use to fully customize the prompts.

            final_answer_as_tool:
                Whether the final output is communicated as a tool call (default is True).
                Disable when your outputs are truncated or low-quality.

            tool_call_checker:
                Configuration for a component that detects a situation when LLM generates tool calls in a cycle.

            notes:
                Additional notes for the agents. The only difference from `instructions` is that notes are at the very
                end of the system prompt and should be more related to the output and its formatting.
        """
        super().__init__(middlewares=middlewares)
        self._llm = ChatModel.from_name(llm) if isinstance(llm, str) else llm
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
                        # pyrefly: ignore [no-matching-overload]
                        "instructions": "\n -".join(cast_list(instructions)) if instructions else None,
                        # pyrefly: ignore [no-matching-overload]
                        "notes": "\n -".join(cast_list(notes)) if notes else None,
                    }
                )
            )
        self._tools = list(tools or [])
        self._requirements = list(requirements or [])
        self._meta = AgentMeta(name=name or "", description=description or "", tools=self._tools)
        self.runner_cls: type[RequirementAgentRunner] = RequirementAgentRunner

    @runnable_entry
    async def run(self, input: str | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]) -> RequirementAgentOutput:
        """Execute the agent.

        Args:
            input: The input to the agent (if list of messages, uses the last message as input)
            expected_output: Pydantic model or instruction for steering the agent towards an expected output format.
            total_max_retries: Maximum number of model retries.
            max_retries_per_step: Maximum number of model retries per step.
            max_iterations: Maximum number of iterations.
            backstory: Additional piece of information or background for the agent.
            signal: The abort signal
            context: A dictionary that can be used to pass additional context to the agent

        Returns:
            The agent output.
        """

        runner = self.runner_cls(
            llm=self._llm,
            config=AgentExecutionConfig(
                max_retries_per_step=kwargs.get("max_retries_per_step", 3),
                total_max_retries=kwargs.get("total_max_retries", 20),
                max_iterations=kwargs.get("max_iterations", 20),
            ),
            tools=self._tools,
            requirements=self._requirements,
            expected_output=kwargs.get("expected_output"),
            tool_call_cycle_checker=self._create_tool_call_checker(),
            run_context=RunContext.get(),
            force_final_answer_as_tool=self._final_answer_as_tool,
            templates=self._templates,
        )
        new_messages = self._process_input(
            input,
            backstory=kwargs.get("backstory"),
            expected_output=kwargs.get("expected_output"),
        )
        await runner.add_messages(self.memory.messages)
        await runner.add_messages(new_messages)

        final_state = await runner.run()

        if self._save_intermediate_steps:
            self.memory.reset()
            await self.memory.add_many(final_state.memory.messages)
        else:
            await self.memory.add_many(new_messages)
            await self.memory.add_many(extract_last_tool_call_pair(final_state.memory) or [])

        assert final_state.answer is not None
        return RequirementAgentOutput(
            output=[final_state.answer],
            output_structured=final_state.result,
            state=final_state,
        )

    def _process_input(
        self, input: str | list[AnyMessage], backstory: str | None, expected_output: Any
    ) -> list[AnyMessage]:
        if not input:
            return []

        *msgs, last_message = [UserMessage(input)] if isinstance(input, str) else input
        if last_message is None and isinstance(last_message, UserMessage) and last_message.text:
            user_message = UserMessage(
                self._templates.task.render(
                    RequirementAgentTaskPromptInput(
                        prompt=last_message.text,
                        context=backstory,
                        expected_output=expected_output if isinstance(expected_output, str) else None,
                    )
                ),
                meta=last_message.meta.copy(),
            )
            user_message.content.extend(
                [content for content in last_message.content if not isinstance(content, MessageTextContent)]
            )
            return [*msgs, user_message]
        else:
            # pyrefly: ignore [bad-return]
            return msgs if last_message is None else [*msgs, last_message]

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
            tool_call_checker=(
                self._tool_call_checker.config.model_copy()
                if isinstance(self._tool_call_checker, ToolCallChecker)
                else self._tool_call_checker
            ),
            save_intermediate_steps=self._save_intermediate_steps,
            final_answer_as_tool=self._final_answer_as_tool,
            name=self._meta.name,
            description=self._meta.description,
            middlewares=self.middlewares.copy(),
        )
        cloned.emitter = await self.emitter.clone()
        cloned.runner_cls = self.runner_cls
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

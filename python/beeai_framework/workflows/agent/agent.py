# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import random
import string
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Self, overload

from pydantic import BaseModel, InstanceOf

from beeai_framework.agents import AnyAgent
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.agents.tool_calling.utils import ToolCallCheckerConfig
from beeai_framework.agents.types import (
    AgentExecutionConfig,
    AgentMeta,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.context import Run
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.readonly_memory import ReadOnlyMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils.dicts import exclude_keys, exclude_none
from beeai_framework.utils.lists import remove_falsy
from beeai_framework.workflows.types import WorkflowRun
from beeai_framework.workflows.workflow import Workflow

AgentFactory = Callable[[ReadOnlyMemory], AnyAgent | Awaitable[AnyAgent]]
AgentWorkflowAgentType = ToolCallingAgent | RequirementAgent


class AgentWorkflowInput(BaseModel):
    input: str
    backstory: str | None = None
    expected_output: str | type[BaseModel] | None = None

    @classmethod
    def from_message(cls, message: AnyMessage) -> Self:
        return cls(input=message.text)

    def to_message(self) -> AssistantMessage:
        text = "\n\nContext:".join(remove_falsy([self.input or "", self.backstory or ""]))
        return AssistantMessage(text)


class Schema(BaseModel):
    inputs: list[InstanceOf[AgentWorkflowInput]]
    current_input: InstanceOf[AgentWorkflowInput] | None = None
    final_answer: str | None = None
    new_messages: list[InstanceOf[AnyMessage]] = []


class AgentWorkflow:
    def __init__(self, name: str = "AgentWorkflow") -> None:
        self._workflow = Workflow(name=name, schema=Schema)

    @property
    def workflow(self) -> Workflow[Schema]:
        return self._workflow

    def run(self, inputs: Sequence[AgentWorkflowInput | AnyMessage]) -> Run[WorkflowRun[Any, Any]]:
        schema = Schema(
            inputs=[
                input if isinstance(input, AgentWorkflowInput) else AgentWorkflowInput.from_message(input)
                for input in inputs
            ],
        )
        return self.workflow.run(schema)

    def del_agent(self, name: str) -> "AgentWorkflow":
        self.workflow.delete_step(name)
        return self

    @overload
    def add_agent(
        self,
        /,
        *,
        name: str | None = None,
        role: str | None = None,
        llm: ChatModel,
        instructions: str | None = None,
        tools: list[InstanceOf[AnyTool]] | None = None,
        execution: AgentExecutionConfig | None = None,
        save_intermediate_steps: bool = True,
        meta: AgentMeta | None = None,
        tool_call_checker: ToolCallCheckerConfig | bool | None = None,
        final_answer_as_tool: bool | None = None,
    ) -> "AgentWorkflow": ...
    @overload
    def add_agent(self, instance: ToolCallingAgent | RequirementAgent, /) -> "AgentWorkflow": ...
    def add_agent(
        self,
        instance: ToolCallingAgent | RequirementAgent | None = None,
        /,
        *,
        name: str | None = None,
        role: str | None = None,
        llm: ChatModel | None = None,
        instructions: str | None = None,
        tools: list[InstanceOf[AnyTool]] | None = None,
        execution: AgentExecutionConfig | None = None,
        save_intermediate_steps: bool = True,
        meta: AgentMeta | None = None,
        tool_call_checker: ToolCallCheckerConfig | bool | None = None,
        final_answer_as_tool: bool | None = None,
    ) -> "AgentWorkflow":
        if instance is None and llm is None:
            raise ValueError("Either instance or the agent configuration must be provided!")

        if not execution:
            execution = AgentExecutionConfig(
                max_retries_per_step=3,
                total_max_retries=3,
                max_iterations=20,
            )

        async def create_agent(memory: BaseMemory) -> ToolCallingAgent | RequirementAgent:
            if instance is not None:
                new_instance = await instance.clone()
                new_instance.memory = memory
                return new_instance

            return ToolCallingAgent(
                llm=llm,  # type: ignore
                tools=tools,
                memory=memory,
                save_intermediate_steps=save_intermediate_steps,
                tool_call_checker=tool_call_checker if tool_call_checker is not None else True,
                final_answer_as_tool=final_answer_as_tool if final_answer_as_tool is not None else True,
                meta=meta
                if meta
                else AgentMeta(
                    name=name or "ToolCallingAgent",
                    description=role if role else instructions if instructions else "helpful agent",
                    tools=tools or [],
                ),
                templates={
                    "system": lambda template: template.update(
                        defaults=exclude_none({"instructions": instructions, "role": role})
                    )
                },
            )

        async def step(state: Schema) -> None:
            memory = UnconstrainedMemory()
            await memory.add_many(state.new_messages)

            last_message = memory.messages[-1].text if memory.messages else ""
            run_input = state.inputs.pop(0).model_copy() if state.inputs else AgentWorkflowInput(input=last_message)
            state.current_input = run_input
            agent = await create_agent(memory.as_read_only())
            run_output = await agent.run(
                run_input.input, **exclude_keys(run_input.model_dump(), {"input"}), **execution.model_dump()
            )

            state.final_answer = run_output.message.text
            if run_input.input:
                state.new_messages.append(UserMessage(run_input.input))
            _memory = run_output.context.get("memory") if run_output.context else None
            state.new_messages.extend(_memory.messages[-2:] if _memory else [])

        self.workflow.add_step(name or f"Agent{''.join(random.choice(string.ascii_letters) for _ in range(4))}", step)
        return self

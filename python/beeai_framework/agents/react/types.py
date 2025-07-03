# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel, InstanceOf

from beeai_framework.agents.react.runners.default.prompts import (
    AssistantPromptTemplateInput,
    SchemaErrorTemplateInput,
    SystemPromptTemplateInput,
    ToolErrorTemplateInput,
    ToolInputErrorTemplateInput,
    ToolNoResultsTemplateInput,
    ToolNotFoundErrorTemplateInput,
    UserEmptyPromptTemplateInput,
    UserPromptTemplateInput,
)
from beeai_framework.agents.types import AgentExecutionConfig, AgentMeta, BaseAgentRunOptions
from beeai_framework.backend import AssistantMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelOutput
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.template import PromptTemplate
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils.strings import to_json


class ReActAgentRunInput(BaseModel):
    prompt: str | None = None


class ReActAgentIterationMeta(BaseModel):
    iteration: int


class ReActAgentRunOptions(BaseAgentRunOptions):
    execution: AgentExecutionConfig | None = None


class ReActAgentIterationResult(BaseModel):
    thought: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    final_answer: str | None = None

    def to_template(self) -> dict[str, str]:
        return {
            "thought": self.thought or "",
            "tool_name": self.tool_name or "",
            "tool_input": to_json(self.tool_input) if self.tool_input else "",
            "tool_output": self.tool_output or "",
            "final_answer": self.final_answer or "",
        }


class ReActAgentRunIteration(BaseModel):
    raw: InstanceOf[ChatModelOutput]
    state: InstanceOf[ReActAgentIterationResult]


class ReActAgentRunOutput(BaseModel):
    result: InstanceOf[AssistantMessage]
    iterations: list[ReActAgentRunIteration]
    memory: InstanceOf[BaseMemory]


class ReActAgentTemplates(BaseModel):
    system: InstanceOf[PromptTemplate[SystemPromptTemplateInput]]
    assistant: InstanceOf[PromptTemplate[AssistantPromptTemplateInput]]
    user: InstanceOf[PromptTemplate[UserPromptTemplateInput]]
    user_empty: InstanceOf[PromptTemplate[UserEmptyPromptTemplateInput]]
    tool_error: InstanceOf[PromptTemplate[ToolErrorTemplateInput]]
    tool_input_error: InstanceOf[PromptTemplate[ToolInputErrorTemplateInput]]
    tool_no_result_error: InstanceOf[PromptTemplate[ToolNoResultsTemplateInput]]
    tool_not_found_error: InstanceOf[PromptTemplate[ToolNotFoundErrorTemplateInput]]
    schema_error: InstanceOf[PromptTemplate[SchemaErrorTemplateInput]]


ReActAgentTemplateFactory = Callable[[InstanceOf[PromptTemplate[Any]]], InstanceOf[PromptTemplate[Any]]]
ReActAgentTemplatesKeys = Annotated[str, lambda v: v in ReActAgentTemplates.model_fields]


class ReActAgentInput(BaseModel):
    llm: InstanceOf[ChatModel]
    tools: list[InstanceOf[AnyTool]]
    memory: InstanceOf[BaseMemory]
    meta: InstanceOf[AgentMeta] | None = None
    templates: dict[ReActAgentTemplatesKeys, InstanceOf[PromptTemplate[Any]] | ReActAgentTemplateFactory] | None = None
    execution: AgentExecutionConfig | None = None
    stream: bool = True

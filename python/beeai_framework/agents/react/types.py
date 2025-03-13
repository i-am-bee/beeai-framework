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

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel, InstanceOf

from beeai_framework.agents.types import AgentExecutionConfig, AgentMeta, BaseAgentRunOptions
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage
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
    result: InstanceOf[AnyMessage]
    iterations: list[ReActAgentRunIteration]
    memory: InstanceOf[BaseMemory]


class ReActAgentTemplates(BaseModel):
    system: InstanceOf[PromptTemplate[Any]]  # TODO proper template subtypes
    assistant: InstanceOf[PromptTemplate[Any]]
    user: InstanceOf[PromptTemplate[Any]]
    user_empty: InstanceOf[PromptTemplate[Any]]
    tool_error: InstanceOf[PromptTemplate[Any]]
    tool_input_error: InstanceOf[PromptTemplate[Any]]
    tool_no_result_error: InstanceOf[PromptTemplate[Any]]
    tool_not_found_error: InstanceOf[PromptTemplate[Any]]
    schema_error: InstanceOf[PromptTemplate[Any]]


ReActAgentTemplateFactory = Callable[[InstanceOf[PromptTemplate[Any]]], InstanceOf[PromptTemplate[Any]]]
ModelKeysType = Annotated[str, lambda v: v in ReActAgentTemplates.model_fields]


class ReActAgentInput(BaseModel):
    llm: InstanceOf[ChatModel]
    tools: list[InstanceOf[AnyTool]]
    memory: InstanceOf[BaseMemory]
    meta: InstanceOf[AgentMeta] | None = None
    templates: dict[ModelKeysType, InstanceOf[PromptTemplate[Any]] | ReActAgentTemplateFactory] | None = None
    execution: AgentExecutionConfig | None = None
    stream: bool | None = None

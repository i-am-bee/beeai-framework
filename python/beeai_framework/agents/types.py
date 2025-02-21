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

import json

from pydantic import BaseModel, InstanceOf

from beeai_framework.backend import Message
from beeai_framework.backend.chat import ChatModel, ChatModelOutput
from beeai_framework.cancellation import AbortSignal
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.templates import PromptTemplate


class BeeRunInput(BaseModel):
    prompt: str | None = None


class BeeMeta(BaseModel):
    iteration: int


class BeeAgentExecutionConfig(BaseModel):
    total_max_retries: int | None = None
    max_retries_per_step: int | None = None
    max_iterations: int | None = None


class BeeRunOptions(BaseModel):
    signal: AbortSignal | None = None
    execution: BeeAgentExecutionConfig | None = None


class BeeIterationResult(BaseModel):
    thought: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: str | None = None
    final_answer: str | None = None

    def to_template(self) -> dict:
        return {
            "thought": self.thought or "",
            "tool_name": self.tool_name or "",
            "tool_input": json.dumps(self.tool_input) if self.tool_input else "",
            "tool_output": self.tool_output or "",
            "final_answer": self.final_answer or "",
        }


class BeeAgentRunIteration(BaseModel):
    raw: InstanceOf[ChatModelOutput]
    state: BeeIterationResult


class BeeRunOutput(BaseModel):
    result: InstanceOf[Message]
    iterations: list[BeeAgentRunIteration]
    memory: InstanceOf[BaseMemory]


class BeeAgentTemplates(BaseModel):
    system: InstanceOf[PromptTemplate]  # TODO proper template subtypes
    assistant: InstanceOf[PromptTemplate]
    user: InstanceOf[PromptTemplate]
    # user_empty: InstanceOf[PromptTemplate]
    # tool_error: InstanceOf[PromptTemplate]
    tool_input_error: InstanceOf[PromptTemplate]
    # tool_no_result_error: InstanceOf[PromptTemplate]
    tool_not_found_error: InstanceOf[PromptTemplate]
    # schema_error: InstanceOf[PromptTemplate]


class AgentMeta(BaseModel):
    name: str
    description: str
    tools: list[InstanceOf[Tool]]
    extra_description: str | None = None


class BeeInput(BaseModel):
    llm: InstanceOf[ChatModel]
    tools: list[InstanceOf[Tool]]  # TODO AnyTool?
    memory: InstanceOf[BaseMemory]
    meta: InstanceOf[AgentMeta] | None = None
    templates: InstanceOf[BeeAgentTemplates] | None = None
    execution: BeeAgentExecutionConfig | None = None
    stream: bool | None = None

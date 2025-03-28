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

from pydantic import BaseModel, Field, InstanceOf

from beeai_framework import AssistantMessage, BaseMemory
from beeai_framework.agents.tool_calling.prompts import (
    ToolCallingAgentSystemPrompt,
    ToolCallingAgentSystemPromptInput,
    ToolCallingAgentTaskPrompt,
    ToolCallingAgentTaskPromptInput,
    ToolCallingAgentToolErrorPrompt,
    ToolCallingAgentToolErrorPromptInput,
)
from beeai_framework.template import PromptTemplate


class ToolCallingAgentTemplates(BaseModel):
    system: InstanceOf[PromptTemplate[ToolCallingAgentSystemPromptInput]] = Field(
        default_factory=lambda: ToolCallingAgentSystemPrompt.fork(None),
    )
    task: InstanceOf[PromptTemplate[ToolCallingAgentTaskPromptInput]] = Field(
        default_factory=lambda: ToolCallingAgentTaskPrompt.fork(None),
    )
    tool_error: InstanceOf[PromptTemplate[ToolCallingAgentToolErrorPromptInput]] = Field(
        default_factory=lambda: ToolCallingAgentToolErrorPrompt.fork(None),
    )


ToolCallingAgentTemplateFactory = Callable[[InstanceOf[PromptTemplate[Any]]], InstanceOf[PromptTemplate[Any]]]
ToolCallingAgentTemplatesKeys = Annotated[str, lambda v: v in ToolCallingAgentTemplates.model_fields]


class ToolCallingAgentRunOutput(BaseModel):
    result: InstanceOf[AssistantMessage]
    memory: InstanceOf[BaseMemory]


class ToolCallingAgentRunState(BaseModel):
    result: InstanceOf[AssistantMessage] | None = None
    memory: InstanceOf[BaseMemory]
    iteration: int

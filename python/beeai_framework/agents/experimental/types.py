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

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from beeai_framework.agents.experimental.prompts import (
    RequirementAgentCycleDetectionPrompt,
    RequirementAgentCycleDetectionPromptInput,
    RequirementAgentSystemPrompt,
    RequirementAgentSystemPromptInput,
    RequirementAgentTaskPrompt,
    RequirementAgentTaskPromptInput,
    RequirementAgentToolErrorPrompt,
    RequirementAgentToolErrorPromptInput,
    RequirementAgentToolNoResultPrompt,
    RequirementAgentToolNoResultTemplateInput,
)
from beeai_framework.agents.experimental.utils._tool import FinalAnswerTool
from beeai_framework.backend import (
    AssistantMessage,
)
from beeai_framework.backend.types import ChatModelToolChoice
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import BaseMemory
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import AnyTool, Tool, ToolOutput


class RequirementAgentTemplates(BaseModel):
    system: InstanceOf[PromptTemplate[RequirementAgentSystemPromptInput]] = Field(
        default_factory=lambda: RequirementAgentSystemPrompt.fork(None),
    )
    task: InstanceOf[PromptTemplate[RequirementAgentTaskPromptInput]] = Field(
        default_factory=lambda: RequirementAgentTaskPrompt.fork(None),
    )
    tool_error: InstanceOf[PromptTemplate[RequirementAgentToolErrorPromptInput]] = Field(
        default_factory=lambda: RequirementAgentToolErrorPrompt.fork(None),
    )
    cycle_detection: InstanceOf[PromptTemplate[RequirementAgentCycleDetectionPromptInput]] = Field(
        default_factory=lambda: RequirementAgentCycleDetectionPrompt.fork(None),
    )
    tool_no_result: InstanceOf[PromptTemplate[RequirementAgentToolNoResultTemplateInput]] = Field(
        default_factory=lambda: RequirementAgentToolNoResultPrompt.fork(None),
    )


RequirementAgentTemplateFactory = Callable[[InstanceOf[PromptTemplate[Any]]], InstanceOf[PromptTemplate[Any]]]
RequirementAgentTemplatesKeys = Annotated[str, lambda v: v in RequirementAgentTemplates.model_fields]


class RequirementAgentRunStateStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    iteration: int
    tool: InstanceOf[Tool[Any, Any, Any]] | None
    input: dict[str, Any]
    output: InstanceOf[ToolOutput]
    error: InstanceOf[FrameworkError] | None


class RequirementAgentRunState(BaseModel):
    result: InstanceOf[AssistantMessage] | None = None
    memory: InstanceOf[BaseMemory]
    iteration: int
    steps: list[RequirementAgentRunStateStep] = []


class RequirementAgentRunOutput(BaseModel):
    result: InstanceOf[AssistantMessage]
    memory: InstanceOf[BaseMemory]
    state: RequirementAgentRunState


class RequirementAgentRequest(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    tools: list[AnyTool]
    allowed_tools: list[AnyTool]
    hidden_tools: list[AnyTool]
    tool_choice: ChatModelToolChoice
    final_answer: FinalAnswerTool
    can_stop: bool

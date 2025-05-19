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

from beeai_framework.agents.controlled.prompts import (
    AbilityAgentAbilityErrorPrompt,
    AbilityAgentAbilityErrorPromptInput,
    AbilityAgentCycleDetectionPrompt,
    AbilityAgentCycleDetectionPromptInput,
    AbilityAgentSystemPrompt,
    AbilityAgentSystemPromptInput,
    AbilityAgentTaskPrompt,
    AbilityAgentTaskPromptInput,
)
from beeai_framework.agents.controlled.requirements.final_answer_tool import FinalAnswerTool
from beeai_framework.backend import (
    AssistantMessage,
)
from beeai_framework.backend.types import ChatModelToolChoice
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import BaseMemory
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import AnyTool, Tool, ToolOutput


class AbilityAgentTemplates(BaseModel):
    system: InstanceOf[PromptTemplate[AbilityAgentSystemPromptInput]] = Field(
        default_factory=lambda: AbilityAgentSystemPrompt.fork(None),
    )
    task: InstanceOf[PromptTemplate[AbilityAgentTaskPromptInput]] = Field(
        default_factory=lambda: AbilityAgentTaskPrompt.fork(None),
    )
    ability_error: InstanceOf[PromptTemplate[AbilityAgentAbilityErrorPromptInput]] = Field(
        default_factory=lambda: AbilityAgentAbilityErrorPrompt.fork(None),
    )
    cycle_detection: InstanceOf[PromptTemplate[AbilityAgentCycleDetectionPromptInput]] = Field(
        default_factory=lambda: AbilityAgentCycleDetectionPrompt.fork(None),
    )


AbilityAgentTemplateFactory = Callable[[InstanceOf[PromptTemplate[Any]]], InstanceOf[PromptTemplate[Any]]]
AbilityAgentTemplatesKeys = Annotated[str, lambda v: v in AbilityAgentTemplates.model_fields]


class AbilityAgentRunStateStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    iteration: int
    tool: InstanceOf[Tool[Any, Any, Any]] | None
    input: dict[str, Any]
    output: InstanceOf[ToolOutput]
    error: InstanceOf[FrameworkError] | None
    # extra: dict[str, Any]  # TODO: stored outputs from Abilities


class AbilityAgentRunState(BaseModel):
    result: InstanceOf[AssistantMessage] | None = None
    memory: InstanceOf[BaseMemory]
    iteration: int
    steps: list[AbilityAgentRunStateStep] = []


class AbilityAgentRunOutput(BaseModel):
    result: InstanceOf[AssistantMessage]
    memory: InstanceOf[BaseMemory]
    state: AbilityAgentRunState


class AbilityAgentRequest(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    tools: list[AnyTool]
    allowed_tools: list[AnyTool]
    hidden_tools: list[AnyTool]
    tool_choice: ChatModelToolChoice
    final_answer: FinalAnswerTool
    can_stop: bool

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

from beeai_framework.agents.governed.prompts import (
    GovernedAgentCycleDetectionPrompt,
    GovernedAgentCycleDetectionPromptInput,
    GovernedAgentSystemPrompt,
    GovernedAgentSystemPromptInput,
    GovernedAgentTaskPrompt,
    GovernedAgentTaskPromptInput,
    GovernedAgentToolErrorPrompt,
    GovernedAgentToolErrorPromptInput,
)
from beeai_framework.agents.governed.utils._tool import FinalAnswerTool
from beeai_framework.backend import (
    AssistantMessage,
)
from beeai_framework.backend.types import ChatModelToolChoice
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import BaseMemory
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import AnyTool, Tool, ToolOutput


class GovernedAgentTemplates(BaseModel):
    system: InstanceOf[PromptTemplate[GovernedAgentSystemPromptInput]] = Field(
        default_factory=lambda: GovernedAgentSystemPrompt.fork(None),
    )
    task: InstanceOf[PromptTemplate[GovernedAgentTaskPromptInput]] = Field(
        default_factory=lambda: GovernedAgentTaskPrompt.fork(None),
    )
    ability_error: InstanceOf[PromptTemplate[GovernedAgentToolErrorPromptInput]] = Field(
        default_factory=lambda: GovernedAgentToolErrorPrompt.fork(None),
    )
    cycle_detection: InstanceOf[PromptTemplate[GovernedAgentCycleDetectionPromptInput]] = Field(
        default_factory=lambda: GovernedAgentCycleDetectionPrompt.fork(None),
    )


GovernedAgentTemplateFactory = Callable[[InstanceOf[PromptTemplate[Any]]], InstanceOf[PromptTemplate[Any]]]
GovernedAgentTemplatesKeys = Annotated[str, lambda v: v in GovernedAgentTemplates.model_fields]


class GovernedAgentRunStateStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    iteration: int
    tool: InstanceOf[Tool[Any, Any, Any]] | None
    input: dict[str, Any]
    output: InstanceOf[ToolOutput]
    error: InstanceOf[FrameworkError] | None


class GovernedAgentRunState(BaseModel):
    result: InstanceOf[AssistantMessage] | None = None
    memory: InstanceOf[BaseMemory]
    iteration: int
    steps: list[GovernedAgentRunStateStep] = []


class GovernedAgentRunOutput(BaseModel):
    result: InstanceOf[AssistantMessage]
    memory: InstanceOf[BaseMemory]
    state: GovernedAgentRunState


class GovernedAgentRequest(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    tools: list[AnyTool]
    allowed_tools: list[AnyTool]
    hidden_tools: list[AnyTool]
    tool_choice: ChatModelToolChoice
    final_answer: FinalAnswerTool
    can_stop: bool

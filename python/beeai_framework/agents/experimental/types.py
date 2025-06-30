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
from typing import Annotated, Any, Generic

from pydantic import BaseModel, ConfigDict, Field, InstanceOf
from typing_extensions import TypeVar

from beeai_framework.agents.experimental.prompts import (
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
    UserMessage,
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
    tool_no_result: InstanceOf[PromptTemplate[RequirementAgentToolNoResultTemplateInput]] = Field(
        default_factory=lambda: RequirementAgentToolNoResultPrompt.fork(None),
    )


RequirementAgentTemplateFactory = Callable[[InstanceOf[PromptTemplate[Any]]], InstanceOf[PromptTemplate[Any]]]
RequirementAgentTemplatesKeys = Annotated[str, lambda v: v in RequirementAgentTemplates.model_fields]


class RequirementAgentRunStateStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    iteration: int
    tool: InstanceOf[Tool[Any, Any, Any]] | None
    input: Any
    output: InstanceOf[ToolOutput]
    error: InstanceOf[FrameworkError] | None


class RequirementAgentRunState(BaseModel):
    answer: InstanceOf[AssistantMessage] | None = None
    result: Any  # TODO
    memory: InstanceOf[BaseMemory]
    iteration: int
    steps: list[RequirementAgentRunStateStep] = []

    @property
    def input(self) -> UserMessage:
        """Get the last user message."""

        return next(msg for msg in reversed(self.memory.messages) if isinstance(msg, UserMessage))


TAnswer = TypeVar("TAnswer", bound=BaseModel, default=Any)


class RequirementAgentRunOutput(BaseModel, Generic[TAnswer]):
    answer: InstanceOf[AssistantMessage]
    answer_structured: TAnswer
    memory: InstanceOf[BaseMemory]
    state: RequirementAgentRunState

    @property
    def result(self) -> AssistantMessage:
        """
        This property is provided for compatibility reasons only.
        Use 'answer' instead.
        """
        return self.answer

    @result.setter
    def result(self, value: AssistantMessage) -> None:
        """
        This setter is provided for compatibility reasons only.
        Sets the 'answer' attribute.
        """
        self.answer = value


class RequirementAgentRequest(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    tools: list[AnyTool]
    allowed_tools: list[AnyTool]
    hidden_tools: list[AnyTool]
    tool_choice: ChatModelToolChoice
    final_answer: FinalAnswerTool
    can_stop: bool

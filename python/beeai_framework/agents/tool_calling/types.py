# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel, Field, InstanceOf

from beeai_framework.agents.tool_calling.prompts import (
    ToolCallingAgentCycleDetectionPrompt,
    ToolCallingAgentCycleDetectionPromptInput,
    ToolCallingAgentSystemPrompt,
    ToolCallingAgentSystemPromptInput,
    ToolCallingAgentTaskPrompt,
    ToolCallingAgentTaskPromptInput,
    ToolCallingAgentToolErrorPrompt,
    ToolCallingAgentToolErrorPromptInput,
)
from beeai_framework.backend import AssistantMessage
from beeai_framework.memory import BaseMemory
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
    cycle_detection: InstanceOf[PromptTemplate[ToolCallingAgentCycleDetectionPromptInput]] = Field(
        default_factory=lambda: ToolCallingAgentCycleDetectionPrompt.fork(None),
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

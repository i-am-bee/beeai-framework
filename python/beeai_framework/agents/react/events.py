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

from typing import Any

from pydantic import BaseModel, InstanceOf

from beeai_framework.agents.react.types import (
    ReActAgentIterationMeta,
    ReActAgentIterationResult,
    ReActAgentRunIteration,
)
from beeai_framework.backend.message import AnyMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.tools.tool import AnyTool


class ReActAgentStartEvent(BaseModel):
    meta: ReActAgentIterationMeta
    tools: list[InstanceOf[AnyTool]]
    memory: InstanceOf[BaseMemory]


class ReActAgentErrorEvent(BaseModel):
    error: InstanceOf[FrameworkError]
    meta: ReActAgentIterationMeta


class ReActAgentRetryEvent(BaseModel):
    meta: ReActAgentIterationMeta


class ReActAgentSuccessEvent(BaseModel):
    data: InstanceOf[AnyMessage]
    iterations: list[ReActAgentRunIteration]
    memory: InstanceOf[BaseMemory]
    meta: ReActAgentIterationMeta


class ReActAgentUpdate(BaseModel):
    key: str
    value: Any
    parsed_value: Any


class ReActAgentUpdateMeta(ReActAgentIterationMeta):
    success: bool


class ReActAgentUpdateEvent(BaseModel):
    data: ReActAgentIterationResult | dict[str, Any]
    update: ReActAgentUpdate
    meta: ReActAgentUpdateMeta
    tools: list[InstanceOf[AnyTool]] | None = None
    memory: InstanceOf[BaseMemory] | None = None


react_agent_event_types: dict[str, type] = {
    "start": ReActAgentStartEvent,
    "error": ReActAgentErrorEvent,
    "retry": ReActAgentRetryEvent,
    "success": ReActAgentSuccessEvent,
    "update": ReActAgentUpdateEvent,
    "partial_update": ReActAgentUpdateEvent,
}

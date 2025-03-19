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

from types import NoneType
from typing import Any

from pydantic import BaseModel, InstanceOf

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.types import ToolOutput, ToolRunOptions


class ToolStartEvent(BaseModel):
    input: InstanceOf[BaseModel]
    options: ToolRunOptions | None = None


class ToolSuccessEvent(BaseModel):
    output: InstanceOf[ToolOutput]
    input: InstanceOf[BaseModel]
    options: ToolRunOptions | None = None


class ToolErrorEvent(BaseModel):
    error: InstanceOf[FrameworkError]
    input: InstanceOf[BaseModel] | dict[str, Any]
    options: ToolRunOptions | None = None


class ToolRetryEvent(BaseModel):
    error: InstanceOf[FrameworkError]
    input: InstanceOf[BaseModel]
    options: ToolRunOptions | None = None


tool_event_types: dict[str, type] = {
    "start": ToolStartEvent,
    "success": ToolSuccessEvent,
    "error": ToolErrorEvent,
    "retry": ToolRetryEvent,
    "finish": NoneType,
}

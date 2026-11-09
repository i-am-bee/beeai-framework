# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from types import NoneType
from typing import Any

from pydantic import BaseModel, InstanceOf, SerializeAsAny

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.types import ToolOutput, ToolRunOptions


class ToolStartEvent(BaseModel):
    input: SerializeAsAny[BaseModel]
    options: ToolRunOptions | None = None


class ToolSuccessEvent(BaseModel):
    output: InstanceOf[ToolOutput]
    input: SerializeAsAny[BaseModel]
    options: ToolRunOptions | None = None


class ToolErrorEvent(BaseModel):
    error: InstanceOf[FrameworkError]
    # Any: the input that failed validation can be any JSON value (for example a list)
    input: SerializeAsAny[BaseModel] | dict[str, Any] | Any
    options: ToolRunOptions | None = None


class ToolRetryEvent(BaseModel):
    error: InstanceOf[FrameworkError]
    input: SerializeAsAny[BaseModel]
    options: ToolRunOptions | None = None


tool_event_types: dict[str, type] = {
    "start": ToolStartEvent,
    "success": ToolSuccessEvent,
    "error": ToolErrorEvent,
    "retry": ToolRetryEvent,
    "finish": NoneType,
}

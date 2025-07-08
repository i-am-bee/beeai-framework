# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from beeai_framework.agents.tool_calling.types import ToolCallingAgentRunState


class ToolCallingAgentStartEvent(BaseModel):
    state: ToolCallingAgentRunState


class ToolCallingAgentSuccessEvent(BaseModel):
    state: ToolCallingAgentRunState


tool_calling_agent_event_types: dict[str, type] = {
    "start": ToolCallingAgentStartEvent,
    "success": ToolCallingAgentSuccessEvent,
}

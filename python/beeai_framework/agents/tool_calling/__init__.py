# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.agents.tool_calling.events import ToolCallingAgentStartEvent, ToolCallingAgentSuccessEvent
from beeai_framework.agents.tool_calling.types import (
    ToolCallingAgentOutput,
    ToolCallingAgentTemplateFactory,
)


# lazy import deprecated aliases to prevent unecessary warnings
def __getattr__(name: str) -> Any:
    if name == "ToolCallingAgentRunOutput":
        from beeai_framework.agents.tool_calling.types import ToolCallingAgentRunOutput

        return ToolCallingAgentRunOutput
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ToolCallingAgent",
    "ToolCallingAgentOutput",
    "ToolCallingAgentRunOutput",
    "ToolCallingAgentStartEvent",
    "ToolCallingAgentSuccessEvent",
    "ToolCallingAgentTemplateFactory",
]

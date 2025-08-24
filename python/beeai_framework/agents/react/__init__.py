# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.events import (
    ReActAgentErrorEvent,
    ReActAgentRetryEvent,
    ReActAgentStartEvent,
    ReActAgentSuccessEvent,
    ReActAgentUpdateEvent,
)
from beeai_framework.agents.react.types import ReActAgentOutput, ReActAgentTemplateFactory


# lazy import deprecated aliases to prevent unecessary warnings
def __getattr__(name: str) -> Any:
    if name == "ReActAgentRunOutput":
        from beeai_framework.agents.react.types import ReActAgentRunOutput

        return ReActAgentRunOutput
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ReActAgent",
    "ReActAgentErrorEvent",
    "ReActAgentOutput",
    "ReActAgentRetryEvent",
    "ReActAgentRunOutput",
    "ReActAgentStartEvent",
    "ReActAgentSuccessEvent",
    "ReActAgentTemplateFactory",
    "ReActAgentUpdateEvent",
]

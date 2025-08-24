# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.adapters.beeai_platform.agents.agent import BeeAIPlatformAgent
from beeai_framework.adapters.beeai_platform.agents.events import (
    BeeAIPlatformAgentErrorEvent,
    BeeAIPlatformAgentUpdateEvent,
)
from beeai_framework.adapters.beeai_platform.agents.types import BeeAIPlatformAgentOutput


def __getattr__(name: str) -> Any:
    if name == "BeeAIPlatformAgentRunOutput":
        from beeai_framework.adapters.beeai_platform.agents.types import BeeAIPlatformAgentRunOutput

        return BeeAIPlatformAgentRunOutput
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BeeAIPlatformAgent",
    "BeeAIPlatformAgentErrorEvent",
    "BeeAIPlatformAgentOutput",
    "BeeAIPlatformAgentRunOutput",
    "BeeAIPlatformAgentUpdateEvent",
]

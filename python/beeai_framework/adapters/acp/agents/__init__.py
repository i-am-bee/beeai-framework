# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.adapters.acp.agents.agent import ACPAgent
from beeai_framework.adapters.acp.agents.events import ACPAgentErrorEvent, ACPAgentUpdateEvent
from beeai_framework.adapters.acp.agents.types import ACPAgentOutput


# lazy import deprecated aliases to prevent unecessary warnings
def __getattr__(name: str) -> Any:
    if name == "ACPAgentRunOutput":
        from beeai_framework.adapters.acp.agents.types import ACPAgentRunOutput

        return ACPAgentRunOutput
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ACPAgent",
    "ACPAgentErrorEvent",
    "ACPAgentOutput",
    "ACPAgentRunOutput",
    "ACPAgentUpdateEvent",
]

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.adapters.a2a.agents.agent import A2AAgent
from beeai_framework.adapters.a2a.agents.events import A2AAgentErrorEvent, A2AAgentUpdateEvent
from beeai_framework.adapters.a2a.agents.types import A2AAgentOutput


# lazy import deprecated aliases to prevent unecessary warnings
def __getattr__(name: str) -> Any:
    if name == "A2AAgentRunOutput":
        from beeai_framework.adapters.a2a.agents.types import A2AAgentRunOutput

        return A2AAgentRunOutput
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "A2AAgent",
    "A2AAgentErrorEvent",
    "A2AAgentOutput",
    "A2AAgentRunOutput",
    "A2AAgentUpdateEvent",
]

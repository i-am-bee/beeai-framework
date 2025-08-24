# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.agents.experimental.agent import RequirementAgent
from beeai_framework.agents.experimental.types import (
    RequirementAgentOutput,
    RequirementAgentRunState,
)


# lazy import deprecated aliases to prevent unecessary warnings
def __getattr__(name: str) -> Any:
    if name == "RequirementAgentRunOutput":
        from beeai_framework.agents.experimental.types import RequirementAgentRunOutput

        return RequirementAgentRunOutput
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["RequirementAgent", "RequirementAgentOutput", "RequirementAgentRunOutput", "RequirementAgentRunState"]

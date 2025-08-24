# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.adapters.watsonx_orchestrate.agents.agent import WatsonxOrchestrateAgent
from beeai_framework.adapters.watsonx_orchestrate.agents.types import WatsonxOrchestrateAgentOutput


# lazy import deprecated aliases to prevent unecessary warnings
def __getattr__(name: str) -> Any:
    if name == "WatsonxOrchestrateAgentRunOutput":
        from beeai_framework.adapters.watsonx_orchestrate.agents.types import WatsonxOrchestrateAgentRunOutput

        return WatsonxOrchestrateAgentRunOutput
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "WatsonxOrchestrateAgent",
    "WatsonxOrchestrateAgentOutput",
    "WatsonxOrchestrateAgentRunOutput",
]

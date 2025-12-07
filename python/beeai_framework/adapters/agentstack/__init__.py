# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"AgentStackAgent", "AgentStackAgentErrorEvent", "AgentStackAgentOutput", "AgentStackAgentUpdateEvent"}:
        import beeai_framework.adapters.agentstack.agents.agent as agent

        warnings.warn(
            f"Please import {name} from beeai_framework.adapters.agentstack.agents.agent instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return getattr(agent, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

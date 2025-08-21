# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from typing import Any, TypeAlias

from beeai_framework.agents import AgentOutput


class WatsonxOrchestrateAgentOutput(AgentOutput):
    raw: dict[str, Any]


# Deprecated: Use 'BeeAIPlatformAgentOutput' instead.
# This alias will be removed in version 0.2
WatsonxOrchestrateAgentRunOutput: TypeAlias = WatsonxOrchestrateAgentOutput

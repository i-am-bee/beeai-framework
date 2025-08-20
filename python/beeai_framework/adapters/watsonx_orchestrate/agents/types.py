# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from beeai_framework.agents import AgentOutput


class WatsonxOrchestrateAgentRunOutput(AgentOutput):
    raw: dict[str, Any]

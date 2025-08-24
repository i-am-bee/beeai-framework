# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING, Any, TypeAlias

from beeai_framework.agents import AgentOutput
from beeai_framework.utils.warnings import deprecated_type_alias


class WatsonxOrchestrateAgentOutput(AgentOutput):
    raw: dict[str, Any]


deprecated_type_alias(__name__, "WatsonxOrchestrateAgentRunOutput", WatsonxOrchestrateAgentOutput)
if TYPE_CHECKING:  # This will only be seen by type checkers
    WatsonxOrchestrateAgentRunOutput: TypeAlias = WatsonxOrchestrateAgentOutput

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, TypeAlias

from beeai_framework.agents import AgentOutput
from beeai_framework.utils.warnings import deprecated_type_alias


class A2AAgentOutput(AgentOutput):
    event: Any


deprecated_type_alias(__name__, "A2AAgentRunOutput", A2AAgentOutput)
if TYPE_CHECKING:  # This will only be seen by type checkers
    A2AAgentRunOutput: TypeAlias = A2AAgentOutput

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, TypeAlias

from acp_sdk.models.models import Event

from beeai_framework.agents import AgentOutput
from beeai_framework.utils.warnings import deprecated_type_alias


class BeeAIPlatformAgentOutput(AgentOutput):
    event: Event


deprecated_type_alias(__name__, "BeeAIPlatformAgentRunOutput", BeeAIPlatformAgentOutput)
if TYPE_CHECKING:  # This will only be seen by type checkers
    BeeAIPlatformAgentRunOutput: TypeAlias = BeeAIPlatformAgentOutput

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

from acp_sdk.models.models import Event

from beeai_framework.agents import AgentOutput


class ACPAgentOutput(AgentOutput):
    event: Event


# Deprecated: Use 'ACPAgentOutput' instead.
# This alias will be removed in version 0.2
ACPAgentRunOutput: TypeAlias = ACPAgentOutput

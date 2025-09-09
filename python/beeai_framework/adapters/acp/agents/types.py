# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from acp_sdk.models.models import Event

from beeai_framework.agents import AgentOutput


class ACPAgentOutput(AgentOutput):
    event: Event

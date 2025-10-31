# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum

try:
    import a2a.client as a2a_client
    import a2a.types as a2a_types
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


from beeai_framework.agents import AgentOutput


class AgentStackAgentOutput(AgentOutput):
    event: a2a_client.ClientEvent | a2a_types.Message


class AgentStackAgentStatus(StrEnum):
    MISSING = "missing"
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"

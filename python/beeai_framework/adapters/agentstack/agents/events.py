# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict

try:
    import a2a.client as a2a_client
    import a2a.types as a2a_types
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


class AgentStackAgentUpdateEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: a2a_client.ClientEvent | a2a_types.Message


class AgentStackAgentErrorEvent(BaseModel):
    message: str


agent_stack_agent_event_types: dict[str, type] = {
    "update": AgentStackAgentUpdateEvent,
    "error": AgentStackAgentErrorEvent,
}

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, ConfigDict

try:
    import a2a.client as a2a_client
    import a2a.types as a2a_types
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e


class A2AAgentUpdateEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: a2a_client.ClientEvent | a2a_types.Message


class A2AAgentErrorEvent(BaseModel):
    message: str


a2a_agent_event_types: dict[str, type] = {
    "update": A2AAgentUpdateEvent,
    "error": A2AAgentErrorEvent,
}

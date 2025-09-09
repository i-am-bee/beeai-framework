# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict

try:
    import a2a.client as a2a_client
    import a2a.types as a2a_types
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e


class BeeAIPlatformAgentUpdateEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: a2a_client.ClientEvent | a2a_types.Message


class BeeAIPlatformAgentErrorEvent(BaseModel):
    message: str


beeai_platform_agent_event_types: dict[str, type] = {
    "update": BeeAIPlatformAgentUpdateEvent,
    "error": BeeAIPlatformAgentErrorEvent,
}

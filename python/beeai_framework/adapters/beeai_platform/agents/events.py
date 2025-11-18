# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel


class BeeAIPlatformAgentUpdateEvent(BaseModel):
    key: str
    value: dict[str, Any]


class BeeAIPlatformAgentErrorEvent(BaseModel):
    message: str


beeai_platform_agent_event_types: dict[str, type] = {
    "update": BeeAIPlatformAgentUpdateEvent,
    "error": BeeAIPlatformAgentErrorEvent,
}

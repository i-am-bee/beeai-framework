# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel


class A2AAgentUpdateEvent(BaseModel):
    value: dict[str, Any]


class A2AAgentErrorEvent(BaseModel):
    message: str


a2a_agent_event_types: dict[str, type] = {
    "update": A2AAgentUpdateEvent,
    "error": A2AAgentErrorEvent,
}

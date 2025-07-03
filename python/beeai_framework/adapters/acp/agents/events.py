# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel


class ACPAgentUpdateEvent(BaseModel):
    key: str
    value: dict[str, Any]


class ACPAgentErrorEvent(BaseModel):
    message: str


acp_agent_event_types: dict[str, type] = {
    "update": ACPAgentUpdateEvent,
    "error": ACPAgentErrorEvent,
}

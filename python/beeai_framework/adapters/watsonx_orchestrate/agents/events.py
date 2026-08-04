# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class WatsonxOrchestrateAgentUpdateEvent(BaseModel):
    delta: str
    content: str


watsonx_orchestrate_agent_event_types: dict[str, type] = {
    "update": WatsonxOrchestrateAgentUpdateEvent,
}

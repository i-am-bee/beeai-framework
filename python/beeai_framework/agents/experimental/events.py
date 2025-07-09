# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from beeai_framework.agents.experimental.types import RequirementAgentRequest, RequirementAgentRunState
from beeai_framework.backend import ChatModelOutput


class RequirementAgentStartEvent(BaseModel):
    state: RequirementAgentRunState
    request: RequirementAgentRequest


class RequirementAgentSuccessEvent(BaseModel):
    state: RequirementAgentRunState
    response: ChatModelOutput


requirement_agent_event_types: dict[str, type] = {
    "start": RequirementAgentStartEvent,
    "success": RequirementAgentSuccessEvent,
}

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from pydantic import BaseModel

from beeai_framework.agents.requirement.types import RequirementAgentRequest, RequirementAgentRunState
from beeai_framework.backend import ChatModelOutput


class RequirementAgentStartEvent(BaseModel):
    state: RequirementAgentRunState
    request: RequirementAgentRequest


class RequirementAgentSuccessEvent(BaseModel):
    state: RequirementAgentRunState
    response: ChatModelOutput


class RequirementAgentFinalAnswerEvent(BaseModel):
    state: RequirementAgentRunState
    output_structured: BaseModel | Any
    output: str
    delta: str


requirement_agent_event_types: dict[str, type] = {
    "start": RequirementAgentStartEvent,
    "success": RequirementAgentSuccessEvent,
    "final_answer": RequirementAgentFinalAnswerEvent,
}

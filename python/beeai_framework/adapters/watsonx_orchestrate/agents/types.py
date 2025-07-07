# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from pydantic import BaseModel, InstanceOf

from beeai_framework.backend import AssistantMessage


class WatsonxOrchestrateAgentRunOutput(BaseModel):
    result: InstanceOf[AssistantMessage]
    raw: dict[str, Any]

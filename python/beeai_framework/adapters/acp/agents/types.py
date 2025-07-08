# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from acp_sdk.models.models import Event
from pydantic import BaseModel, InstanceOf

from beeai_framework.backend.message import AnyMessage


class ACPAgentRunOutput(BaseModel):
    result: InstanceOf[AnyMessage]
    event: Event

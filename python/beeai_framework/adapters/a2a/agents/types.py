# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel, InstanceOf

from beeai_framework.backend.message import AnyMessage


class A2AAgentRunOutput(BaseModel):
    result: InstanceOf[AnyMessage]
    event: Any

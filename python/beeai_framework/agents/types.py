# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, InstanceOf

from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils import AbortSignal


class BaseAgentRunOptions(BaseModel):
    signal: AbortSignal | None = None


class AgentExecutionConfig(BaseModel):
    total_max_retries: int | None = None
    max_retries_per_step: int | None = None
    max_iterations: int | None = None


class AgentMeta(BaseModel):
    name: str
    description: str
    tools: list[InstanceOf[AnyTool]]
    extra_description: str | None = None

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel

from beeai_framework.backend import Role


class OpenAIEvent(BaseModel):
    role: Role = Role.ASSISTANT
    text: str = ""
    finish_reason: str | None = None

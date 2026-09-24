# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field


class EventTrace(BaseModel):
    id: str
    run_id: str
    parent_run_id: str | None = None


class EmitterOptions(BaseModel):
    is_blocking: bool | None = None
    once: bool | None = None
    persistent: bool | None = None
    match_nested: bool | None = None
    priority: int = Field(
        default=0,
        description="Defines the priority in which the callback gets executed. A higher value means earlier execution.",
    )

    model_config = ConfigDict(frozen=True)

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict


class EventTrace(BaseModel):
    id: str
    run_id: str
    parent_run_id: str | None = None


class EmitterOptions(BaseModel):
    is_blocking: bool | None = None
    once: bool | None = None
    persistent: bool | None = None
    match_nested: bool | None = None

    model_config = ConfigDict(frozen=True)

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, InstanceOf

from beeai_framework.tools import AnyTool


class RequirementInitEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    tools: list[InstanceOf[AnyTool]]


requirement_event_types: dict[str, type] = {"init": RequirementInitEvent}

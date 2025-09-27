# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class WorkflowStartEvent(BaseModel): ...


class WorkflowSuccessEvent(BaseModel): ...


workflow_event_types: dict[str, type] = {
    "start": WorkflowStartEvent,
    "success": WorkflowSuccessEvent,
}

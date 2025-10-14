# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict

from beeai_framework.workflows.v2.step import WorkflowStep


class WorkflowStartEvent(BaseModel): ...


class WorkflowStepEvent(BaseModel):
    step: WorkflowStep
    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowStartStepEvent(WorkflowStepEvent):
    pass


class WorkflowErrorEvent(WorkflowStepEvent):
    error: Exception
    attempt: int


class WorkflowRetryStepEvent(WorkflowStepEvent):
    error: Exception
    attempt: int


class WorkflowSuccessEvent(BaseModel): ...


workflow_v2_event_types: dict[str, type] = {
    "start": WorkflowStartEvent,
    "start_step": WorkflowStartStepEvent,
    "error": WorkflowErrorEvent,
    "retry_step": WorkflowRetryStepEvent,
    "success": WorkflowSuccessEvent,
}

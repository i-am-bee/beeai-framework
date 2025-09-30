# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from pydantic import BaseModel, ConfigDict

from beeai_framework.workflows.v2.step import WorkflowStep


class StartWorkflowEvent(BaseModel): ...


class WorkflowStepEvent(BaseModel):
    step: WorkflowStep
    model_config = ConfigDict(arbitrary_types_allowed=True)


class StartWorkflowStepEvent(WorkflowStepEvent):
    pass


class ErrorWorkflowEvent(WorkflowStepEvent):
    error: Exception
    attempt: int


class RetryWorkflowStepEvent(WorkflowStepEvent):
    error: Exception
    attempt: int


class WorkflowSuccessEvent(BaseModel): ...


class WorkflowEventNames(str, Enum):
    START_WORKFLOW = "start"
    START_WORKFLOW_STEP = "start_step"
    WORKFLOW_ERROR = "error"
    RETRY_WORKFLOW_STEP = "retry_step"
    WORKFLOW_SUCCESS = "success"


workflow_event_types: dict[str, type] = {
    WorkflowEventNames.START_WORKFLOW: StartWorkflowEvent,
    WorkflowEventNames.START_WORKFLOW_STEP: StartWorkflowStepEvent,
    WorkflowEventNames.WORKFLOW_ERROR: ErrorWorkflowEvent,
    WorkflowEventNames.RETRY_WORKFLOW_STEP: RetryWorkflowStepEvent,
    WorkflowEventNames.WORKFLOW_SUCCESS: WorkflowSuccessEvent,
}

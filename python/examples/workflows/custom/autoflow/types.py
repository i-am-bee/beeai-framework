from typing import Any

from pydantic import BaseModel, InstanceOf

from beeai_framework.memory import BaseMemory
from beeai_framework.workflows import WorkflowHandler


class AutoflowState(BaseModel):
    task: str
    context: str
    memory: InstanceOf[BaseMemory]
    last_step: str | None = None
    last_input: dict[str, Any] | None
    last_result: Any
    next_step: str


AutoflowHandler = WorkflowHandler[..., str]  # type: ignore[misc]


class AutoflowStep(BaseModel):
    name: str
    description: str
    handler: AutoflowHandler
